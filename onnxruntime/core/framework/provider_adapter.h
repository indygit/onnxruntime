// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {
class ExecutionProviderAdapter : public IExecutionProvider {
public:
  ExecutionProviderAdapter(OrtExecutionProvider* ep) : IExecutionProvider(ep->type), ep_impl_(ep) {}
  virtual std::vector<std::unique_ptr<ComputeCapability>> GetCapability(const GraphViewer& graph_viewer, const IKernelLookup& kernel_lookup) const override {
    size_t cnt = 0;
    OrtIndexedSubGraph** indexed_subgraph = nullptr;
    ep_impl_->GetCapability(ep_impl_, reinterpret_cast<const OrtGraphViewer*>(&graph_viewer), &cnt, &indexed_subgraph);

    if (cnt == 0) return IExecutionProvider::GetCapability(graph_viewer, kernel_lookup);

    std::vector<std::unique_ptr<ComputeCapability>> ret;
    for (size_t i = 0; i < cnt; i++) {
        std::unique_ptr<IndexedSubGraph> sb = std::make_unique<IndexedSubGraph>();
        sb->nodes.reserve(indexed_subgraph[i]->node_index_len);
        for (size_t j = 0; j < indexed_subgraph[i]->node_index_len; j++) sb->nodes.push_back((indexed_subgraph[i]->node_index)[j]);
        if (indexed_subgraph[i]->meta_def != nullptr) {
            std::unique_ptr<IndexedSubGraph::MetaDef> meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
            meta_def->name = indexed_subgraph[i]->meta_def->name ? indexed_subgraph[i]->meta_def->name : "";
            meta_def->doc_string = indexed_subgraph[i]->meta_def->doc_string ? indexed_subgraph[i]->meta_def->doc_string : "";
            meta_def->domain = indexed_subgraph[i]->meta_def->domain ? indexed_subgraph[i]->meta_def->domain : "";
            meta_def->since_version = indexed_subgraph[i]->meta_def->since_version;

            meta_def->inputs.reserve(indexed_subgraph[i]->meta_def->input_len);
            for (size_t j = 0; j < indexed_subgraph[i]->meta_def->input_len; j++) meta_def->inputs.push_back(indexed_subgraph[i]->meta_def->inputs[j]);

            meta_def->outputs.reserve(indexed_subgraph[i]->meta_def->output_len);
            for (size_t j = 0; j < indexed_subgraph[i]->meta_def->output_len; j++) meta_def->outputs.push_back(indexed_subgraph[i]->meta_def->outputs[j]);

            meta_def->constant_initializers.reserve(indexed_subgraph[i]->meta_def->initializer_len);
            for (size_t j = 0; j < indexed_subgraph[i]->meta_def->initializer_len; j++) meta_def->constant_initializers.push_back(indexed_subgraph[i]->meta_def->constant_initializers[j]);

            sb->SetMetaDef(std::move(meta_def));
        }

        ret.push_back(std::make_unique<ComputeCapability>(std::move(sb)));
    }
    return ret;
  }

  virtual common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs, std::vector<NodeComputeInfo>& node_compute_funcs) override {
    std::vector<const OrtGraphViewer*> ortGraphs;
    std::vector<const OrtNode*> ortNodes;
    for (auto& fused_node_graph : fused_nodes_and_graphs) {
      const GraphViewer& graph_viewer = fused_node_graph.filtered_graph;
      const Node& fused_node = fused_node_graph.fused_node;
      ortGraphs.push_back(reinterpret_cast<const OrtGraphViewer*>(&graph_viewer));
      ortNodes.push_back(reinterpret_cast<const OrtNode*>(&fused_node));
    }
    size_t count = fused_nodes_and_graphs.size();
    node_compute_info_ = new OrtNodeComputeInfo* [count];
    ep_impl_->Compile(ep_impl_, ortGraphs.data(), ortNodes.data(), count, &node_compute_info_);

    node_compute_funcs.reserve(count);
    for (size_t i = 0; i < count; i++) {
        NodeComputeInfo compute_info;
        compute_info.create_state_func = [&](ComputeContext* context, void** state) {
            if (node_compute_info_[0]->CreateFunctionStateFunc) {
                OrtComputeContext occ;
                occ.AllocateFunc = context->allocate_func;
                occ.DestroyFunc = context->release_func;
                occ.allocator_handle = context->allocator_handle;
                occ.node_name = context->node_name;
                return node_compute_info_[0]->CreateFunctionStateFunc(&occ, state);  // TODO(leca): reinterpret_cast<OrtComputeContext*>(context)?
            }
            return 0;
        };
        compute_info.compute_func = [&](void* state, const OrtApi* api, OrtKernelContext* context) {
            return ToStatus(node_compute_info_[0]->ComputeFunc(state, api, context));
        };
        compute_info.release_state_func = [&](void* state) {
            if (node_compute_info_[0]->DestroyFunctionStateFunc) {
                node_compute_info_[0]->DestroyFunctionStateFunc(state);
            }
        };
        node_compute_funcs.push_back(compute_info);
    }
    return Status::OK();
  }
private:
  OrtExecutionProvider* ep_impl_;
  OrtNodeComputeInfo** node_compute_info_;
};
}
