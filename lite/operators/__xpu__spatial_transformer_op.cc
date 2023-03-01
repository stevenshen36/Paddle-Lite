

// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/operators/__xpu__spatial_transformer_op.h"
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

static std::vector<std::vector<int>> vec1DTo2D_int(const std::vector<int>& vec,
                                                   int dim) {
  std::vector<std::vector<int>> res;
  for (size_t i = 0; i < vec.size(); i += dim) {
    std::vector<int> tmp;
    for (size_t j = 0; j < dim; j++) {
      tmp.push_back(vec[i + j]);
    }
    res.emplace_back(std::move(tmp));
  }
  return res;
}

bool XPUSpatialTransformerOp::CheckShape() const {
  CHECK_EQ(param_.input->dims().size(), 4UL);
  return true;
}

bool XPUSpatialTransformerOp::InferShapeImpl() const {
  auto input_shape = param_.input->dims();
  auto batch_size = input_shape[0];
  auto channel = input_shape[1];
  auto height = input_shape[2];
  auto weight = input_shape[3];
  param_.output->Resize({batch_size, channel, height, weight});
  return true;
}

bool XPUSpatialTransformerOp::AttachImpl(const cpp::OpDesc& op_desc,
                                         lite::Scope* scope) {
  param_.input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.embedding =
      scope->FindVar(op_desc.Input("Embedding").front())->GetMutable<Tensor>();
  param_.output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();

  param_.fc_weight.clear();
  for (auto& name : op_desc.Input("FCWeight")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.fc_weight.push_back(t);
  }
  param_.fc_bias.clear();
  for (auto& name : op_desc.Input("FCBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.fc_bias.push_back(t);
  }
  param_.ln_scale.clear();
  for (auto& name : op_desc.Input("LNScale")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.ln_scale.push_back(t);
  }
  param_.ln_bias.clear();
  for (auto& name : op_desc.Input("LNBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.ln_bias.push_back(t);
  }
  param_.conv_weight.clear();
  for (auto& name : op_desc.Input("ConvWeight")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.conv_weight.push_back(t);
  }
  param_.conv_bias.clear();
  for (auto& name : op_desc.Input("ConvBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.conv_bias.push_back(t);
  }
  param_.gn_scale.clear();
  for (auto& name : op_desc.Input("GNScale")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.gn_scale.push_back(t);
  }
  param_.gn_bias.clear();
  for (auto& name : op_desc.Input("GNBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.gn_bias.push_back(t);
  }

  param_.hidden_dim = op_desc.GetAttr<int>("hidden_dim");
  param_.head_num = op_desc.GetAttr<int>("head_num");
  param_.size_per_head = op_desc.GetAttr<int>("size_per_head");
  param_.embedding_dim = op_desc.GetAttr<int>("embedding_dim");
  param_.geglu_dim = op_desc.GetAttr<int>("gelu_dim");
  param_.groups = op_desc.GetAttr<int>("groups");
  param_.epsilon = op_desc.GetAttr<float>("epsilon");

  param_.weight_max.clear();
  for (const auto& weight_max_tensor :
       op_desc.GetAttr<std::vector<std::string>>("FCWeightMax")) {
    auto tensor = scope->FindMutableTensor(weight_max_tensor);
    CHECK(tensor != nullptr);
    param_.weight_max.push_back(tensor);
  }
  param_.conv_max.clear();
  for (const auto& weight_max_tensor :
       op_desc.GetAttr<std::vector<std::string>>("ConvFilterMax")) {
    auto tensor = scope->FindMutableTensor(weight_max_tensor);
    CHECK(tensor != nullptr);
    param_.conv_max.push_back(tensor);
  }
  param_.conv_groups = op_desc.GetAttr<std::vector<int>>("Conv_Groups");
  param_.strides =
      vec1DTo2D_int(op_desc.GetAttr<std::vector<int>>("Strides"), 2);
  param_.paddings =
      vec1DTo2D_int(op_desc.GetAttr<std::vector<int>>("Paddings"), 4);
  param_.dilations =
      vec1DTo2D_int(op_desc.GetAttr<std::vector<int>>("Dilations"), 2);
  param_.filter_dims =
      vec1DTo2D_int(op_desc.GetAttr<std::vector<int>>("FilterDims"), 4);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__spatial_transformer,
                 paddle::lite::operators::XPUSpatialTransformerOp);
