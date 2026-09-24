//
//  SinatraNet.swift
//  SinatraMLX
//
//  WHAT: The time-series side model. Per retrieved partition it reads the 38 time and
//        indicator features plus a 64-d sketch of the partition's pooled embedding, and
//        emits a weight w_p ∈ [−1, 1] (how hard to steer decoding toward that partition)
//        and an engagement forecast R̂ ∈ [0, 1].
//
//          ctx_proj: 64 → 8 ──┐
//          features (34) ─────┴→ input: 42 → 32 → GELU → hidden: 32 → 16 → GELU
//                                   ├→ weight_head: 16 → 1 → tanh      (zero-initialised: w = 0 at birth)
//                                   └→ gate_head:   16 → 1 → sigmoid   (forecast)
//
//  PIN:  ~2k parameters, ~10 KB on disk: sized for the few hundred rows a 30-day band
//        holds. v2 note: replace `hidden` with attention across the turn's partitions so
//        they compete for the injection.
//

import Foundation
import MLX
import MLXNN

final class SinatraNet: Module {
    @ModuleInfo(key: "ctx_proj") var ctxProj: Linear
    @ModuleInfo(key: "input") var input: Linear
    @ModuleInfo(key: "hidden") var hidden: Linear
    @ModuleInfo(key: "weight_head") var weightHead: Linear
    @ModuleInfo(key: "gate_head") var gateHead: Linear
    /// Input dropout while training: no single drifting feature can carry the prediction.
    let featureDropout = Dropout(p: 0.2)
    let contextDropout = Dropout(p: 0.1)

    init(featureCount: Int, contextDim: Int) {
        self._ctxProj.wrappedValue = Linear(contextDim, 8)
        self._input.wrappedValue = Linear(featureCount + 8, 32)
        self._hidden.wrappedValue = Linear(32, 16)
        self._weightHead.wrappedValue = Linear(weight: MLXArray.zeros([1, 16]), bias: MLXArray.zeros([1]))
        self._gateHead.wrappedValue = Linear(16, 1)
        super.init()
    }

    func callAsFunction(features: MLXArray, context: MLXArray) -> (weights: MLXArray, forecast: MLXArray) {
        let c = ctxProj(contextDropout(context))
        let x = concatenated([featureDropout(features), c], axis: -1)
        let h = gelu(hidden(gelu(input(x))))
        let w = tanh(weightHead(h)).squeezed(axis: -1)
        let r = sigmoid(gateHead(h)).squeezed(axis: -1)
        return (w, r)
    }
}
