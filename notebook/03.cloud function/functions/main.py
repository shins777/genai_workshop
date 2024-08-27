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

import os
from flask import Flask, request, json
from rag.controller import Controller
import functions_framework

app = Flask(__name__)


# Production env.
prod = True

@functions_framework.http
def search(request):

    params = request.get_json()
    
    question = params['question']
    print(f" question : {question}")

    controller = Controller(prod)
    outcome = controller.rag_process( question = question, top_n = 5  )

    print(f"result {outcome}")

    response = {
        "result": outcome
    }

    return json.dumps(response,ensure_ascii=False)
