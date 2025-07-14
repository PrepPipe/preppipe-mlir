#ifndef PREPPIPE_VNMODEL_BASE_TD
#define PREPPIPE_VNMODEL_BASE_TD

include "mlir/IR/OpBase.td"

def VNModel_Dialect : Dialect {
  let name = "vnmodel";
  let cppNamespace = "::mlir::preppipe::VNModel";
  let summary = "Abstract Visual Novel Content Model Dialect";
  let description = [{
    Visual novel abstract content representation dialect for PrepPipe Compiler.
    语涵编译器的视觉小说抽象内容表示方言。

    VNModel is the engine-agnostic intermediate representation for visual novel
    contents. It contains imperative commands that encodes all the aspects of
    visual novel content and staging, but is designed to be friendly to
    analysis and transforms. Third-party frontends should generate IR in
    VNModel dialect to use PrepPipe's lowering pipeline.
    VNModel 是视觉小说内容的引擎无关中间表示(IR)。它使用命令来描述视觉小说的内容和
    演出，同时设计趋向于便于分析和转换。第三方前端应该生成 VNModel 下的 IR 来使用
    语涵编译器的下沉管线。
  }];
}

#endif // PREPPIPE_VNMODEL_BASE_TD
