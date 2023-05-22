/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common.hpp"
#include "kernel_func.hpp"
#include "utils/utils.hpp"

using namespace std::placeholders;

TEST(block_load_store, esimd) {
    cl::sycl::nd_range<1> Range({1}, {1});
    auto result_validate = std::bind(
            block_load_store_result_validate, _1, _2, _3, 32, 16, 16);
    kernel_run<int, block_load_store_func<int, 32, 32, 32, 16, 16>>(
            Range, result_validate);
}
