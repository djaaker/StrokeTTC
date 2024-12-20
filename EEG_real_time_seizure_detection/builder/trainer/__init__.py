# Copyright (c) 2022, Kwanhyung Lee. All rights reserved.
#
# Licensed under the MIT License; 
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Remove wildcard import
# from .trainer import *
from .OurVers_trainer import sliding_window_v2

# You can also add other imports here as needed


def get_trainer(args, iteration, x, y, target_lengths, model, logger, device, scheduler, optimizer, criterion, signal_name_list, flow_type="train"):
    model, iter_loss = sliding_window_v2(args, iteration, x, y, target_lengths, model, logger, device, scheduler, optimizer, criterion, signal_name_list, flow_type)

    return model, iter_loss