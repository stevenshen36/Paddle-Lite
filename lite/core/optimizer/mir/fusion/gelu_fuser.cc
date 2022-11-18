// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/fusion/gelu_fuser.h"
#include <cmath>
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void QuickGELUFuser::BuildPattern() {

  auto scale_teller = [](const Node* node) -> bool {
    bool bias_after_scale =
        const_cast<Node*>(node)->AsStmt().op_info()->GetAttr<bool>(
            "bias_after_scale");
    bool has_act =
        const_cast<Node*>(node)->AsStmt().op_info()->HasAttr("activation_type");
    return bias_after_scale && (!has_act);
  };

  /*                 _____________________  
                    /                     \
    Create node: X----scale----sigmoid---elementwise_mul---output
  */
  auto* x = VarNode("x")->assert_is_op_input("scale", "X");
  auto* scale = OpNode("scale", "scale")
                ->assert_is_op("scale")
                ->assert_node_satisfied(scale_teller);
  auto* scale_out = VarNode("scale_out");
  auto* sigmoid = OpNode("sigmoid", act_type_);
  auto* sigmoid_out = VarNode("sigmoid_out");
  auto* element_mul = OpNode("elementwise_mul",op_type_)
                      ->assert_op_attr_satisfied<int>(
                          "axis",[](int attr){return attr == -1 || attr == 0;});
  auto* output = VarNode("Out");

  // Construct the topological structure for scale-sigmoid-elementwise_mul
  *x >> *scale >> *scale_out >> *sigmoid >> *sigmoid_out;
  std::vector<PMNode*> element_mul_inputs{x, sigmoid_out};
  element_mul_inputs >> *element_mul >> *output;

  // Some op specialities.
  scale->AsIntermediate();
  scale_out->AsIntermediate();
  sigmoid->AsIntermediate();
  sigmoid_out->AsIntermediate();
  element_mul->AsIntermediate();

  return;
}

void QuickGELUFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  // get op_desc for gelu op.
  auto op_desc = GenOpDesc(matched);
  // Create gelu op.
  auto gelu_op = LiteOpRegistry::Global().Create("quick_gelu");

  // find scope and valid_places of original scale op.
  auto scale = matched.at("scale")->stmt()->op();
  auto* scope = scale->scope();
  auto& valid_places = scale->valid_places();

  // set gelu op's scope and valid_places which aligned with scale op.
  gelu_op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(gelu_op, valid_places);

  // link IO to the new op node.
  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
}

cpp::OpDesc QuickGELUFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("scale")->stmt()->op_info();
  float scale_val = op_desc.GetAttr<float>("scale");
  op_desc.mutable_inputs()->clear();
  op_desc.mutable_outputs()->clear();
  op_desc.SetType("quick_gelu");
  op_desc.SetInput("X", {matched.at("x")->arg()->name});
  op_desc.SetOutput("Out",{matched.at("Out")->arg()->name});
  op_desc.SetAttr("scale", scale_val);
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
