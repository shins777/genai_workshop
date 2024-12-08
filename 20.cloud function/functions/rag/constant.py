# Copyright 2024 shins777@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Logging level( INFO | DEBUG )
logging = "INFO"

#  Project information
project_id = "ai-hangsik"
region="asia-northeast3"

model = "gemini-1.5-flash"

# if svc acct file exists.
#svc_acct_file = "/home/admin_/keys/ai-hangsik-71898c80c9a5.json"

search_url = "https://discoveryengine.googleapis.com/v1alpha/projects/721521243942/locations/global/collections/default_collection/dataStores/it-laws-ds_1713063479348/servingConfigs/default_search:search"

ranker_model = f"semantic-ranker-512@latest"
rank_score = 0.3
num_search = 3
num_question = 3
