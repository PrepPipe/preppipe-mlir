#ifndef PREPPIPE_VNMODEL_OPS_TD
#define PREPPIPE_VNMODEL_OPS_TD

class VNModel_Op<string mnemonic, list<Trait> traits = []>
    : Op<VNModel_Dialect, mnemonic, traits> {
}

// def VNModel_ModuleOp : VNModel_Op<"module", [// TODO
//     DeclareOpInterfaceMethods<SymbolUserOpInterface>,
//     SingleBlock, NoTerminator]> {
//   let summary = "Top-level module operation for PrepPipe's VNModel dialect";
//   let description = [{
//     This is the top-level module operation used in the VNModel level in
//     PrepPipe Compiler's visual novel lowering pipeline. Currently, there
//     should be only one single such operation for an entire visual novel.
//     In the future, if DLC support is added, the top-level module would correspond
//     to the base game, and there would be additional module operations for
//     each DLC.
//     这是语涵编译器视觉小说下沉管线中的顶层模块 Operation 。目前整个视觉小说中应该
//     只有一个这样的 Operation 。未来支持 DLC 时，顶层模块将对应于游戏的基础部分，
//     而每个 DLC 将有一个额外的模块。
//   }];

//   let arguments = (ins);
//   //let results = (outs Torch_NnModuleType:$result);
//   let regions = (region SizedRegion<1>:$region);
//   //let hasVerifier = 1;

//   let assemblyFormat = "$region attr-dict `:` qualified(type($result))";
// }

#endif // PREPPIPE_VNMODEL_OPS_TD
