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
import time
import ast
import requests
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from operator import is_not
from functools import partial

import google
import google.oauth2.credentials
import google.auth.transport.requests
from google.oauth2 import service_account

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from google.cloud import discoveryengine_v1alpha as discoveryengine

import rag.constant as env
import logging

logging.basicConfig(
  format = '%(asctime)s:%(levelname)s:%(message)s',
  level = logging.INFO
)


class Controller():
  
  credentials = None
  re_rank_client = None
  ranking_config = None

  def __init__(self, prod:bool ):

      # Logger setting. 
      if env.logging == "INFO": logging.getLogger().setLevel(logging.INFO)
      else: logging.getLogger().setLevel(logging.DEBUG)

      # if prod = False, use svc account.
      if not prod:
          # the location of service account in Cloud Shell.
          Controller.credentials = service_account.Credentials.from_service_account_file(
              env.svc_acct_file, 
              scopes=['https://www.googleapis.com/auth/cloud-platform']
          )
      else:
          # Use default auth in Cloud Run env. 
          Controller.credentials, project_id = google.auth.default()

      Controller.re_rank_client = discoveryengine.RankServiceClient()

      Controller.ranking_config = Controller.re_rank_client.ranking_config_path(
          project=env.project_id,
          location="global",
          ranking_config="default_ranking_config",
      )

      # Initialize Vertex AI env with the credentials. 
      vertexai.init(project=env.project_id, location=env.region, credentials = Controller.credentials )

      logging.info(f"[__init__] Controller.credentials  : {Controller.credentials}")
      logging.info(f"[__init__] Controller.gemini_model  : {env.model}")
      logging.info(f"[__init__] Initialize Controller done!")

  def rag_process(self, question:str, top_n: int = 5 ):
      """
      Controller to execute the RAG processes.
      """

      print(f"[rag_process] start rag_process : {question}")

      t1 = time.perf_counter()
      execution_stat = {}

      rewrited_questions = self.rewrite_question(question)
      rewrited_questions.append(question) # Add original question
      print(f"[rag_process] rewrited_questions : {rewrited_questions}")

      t2 = time.perf_counter()

      searched_list = self.search_contexts(rewrited_questions)
      print(f"[rag_process] searched_list : {searched_list}")

      t3 = time.perf_counter()

      ranked_results = self.ranking_results(question, searched_list, top_n)
      print(f"[rag_process] ranked_results : {ranked_results}")

      t4 = time.perf_counter()

      final_contexts = "\n Searched Context : ".join(ranked_results)

      final_prompt = f"""
        You are an AI assistant that searches for knowledge and provides some advice.
        When answering the <Question> below, please refer only to the contents within <Context>, infer step by step, summarize, and answer.

        <Context>{final_contexts}</Context>
        <Question>{question}</Question>
      """

      final_outcome = self.gemini_response(final_prompt)
      t5 = time.perf_counter()

      execution_stat['query_rewrite'] = round((t2-t1), 3)
      execution_stat['vertex_ai_search'] = round((t3-t2), 3)
      execution_stat['ranked_results'] = round((t4-t3), 3)
      execution_stat['llm_request'] = round((t5-t4), 3)

      return final_outcome, execution_stat

  #----------------------------------------------------------------------------------------------------------------

  def rewrite_question(self, question:str )->list:

    prompt = f"""
      This is a question generator for your precise search.
      In order to search for facts to answer the [Question] below, please create 3 questions based on [Question].
      Your answers must be in the list format below.

        [Question] : {question}
        Format : ["Question 1", "Question 2", "Question 3"]

    """
    questions = self.gemini_response(prompt)
    print(f"[Controller][rewrite_question] questions : {questions}")

    q_list = []

    try:
        q_list = ast.literal_eval(questions)

    # Handling for exception when splitting mixed question.
    except Exception as e:
        print(f"[Controller][rewrite_question] Query rewrite failed")
        for i in range(env.num_question):
            q_list.append(question)

    print(f"[Controller][rewrite_question] Generated Question List : {q_list}")

    return q_list


  #----------------------------------------------------------------------------------------------------------------

  def search_contexts(self, rewrited_questions):
      """
      Controller to execute the RAG processes.

      1. Call flow for mixed question:
          question_splitter --> search_chunks
      2. Call flow for singuar question:
          search_chunks
      """

      # t1 = time.perf_counter()

      # Parallel processing to reduce the latency for the Vertex AI Search.
      with ThreadPoolExecutor(max_workers=10) as executor:
          searched_contexts = executor.map(self.search_parsing, rewrited_questions )

      searched_list = [context for context in searched_contexts]

      print(f"[Controller][search] len(searched_list) : {len(searched_list)}")
      print(f"[Controller][search] searched_list : {searched_list}")

      return searched_list



  #----------------------------------------------------------------------------------------------

  def search_parsing(self, question:str)->str:
      print(f"[Controller][search_parsing] search_parsing Start! : {question}")

      #------- Searching --------------------------------------------------------------------
      request = google.auth.transport.requests.Request()
      Controller.credentials.refresh(request)

      headers = {
          "Authorization": "Bearer "+ Controller.credentials.token,
          "Content-Type": "application/json"
      }

      query_dic ={
          "query": question,
          "page_size": str(env.num_search),
          "offset": 0,
          "contentSearchSpec":{
                  "searchResultMode" : "CHUNKS",
                  "chunkSpec" : {
                      "numPreviousChunks" : 1,
                      "numNextChunks" : 1
                  }
          },
      }

      data = json.dumps(query_dic)
      data=data.encode("utf8")
      response = requests.post(env.search_url,headers=headers, data=data)

      print(f"[Controller][search_parsing] Search Response len : {len(response.text)}")
      #print(f"[Controller][search_parsing] Search Response chunks : {response.text}")
      print(f"[Controller][search_parsing] Search End! : {question}")

      # Start to parse the searched chunks
      dict_results = json.loads(response.text)

      #------- Parsing --------------------------------------------------------------------

      search_results = {}

      if dict_results.get('results'):
          for result in dict_results['results']:
              item = {}
              chunk = result['chunk']
              item['title'] = chunk['documentMetadata']['title']
              item['uri'] = chunk['documentMetadata']['uri']
              item['pageSpan'] = f"{chunk['pageSpan']['pageStart']} ~ {chunk['pageSpan']['pageEnd']}"
              item['content'] = chunk['content']
              item['question'] = question

              if 'chunkMetadata' in chunk:
                  add_chunks = chunk['chunkMetadata']
                  if 'previousChunks' in add_chunks:
                      # Chunks appearing from those closest to the current Contents.
                      p_chunks = chunk['chunkMetadata']['previousChunks']
                      if p_chunks:
                          for p_chunk in p_chunks:
                              item['content'] = p_chunk['content'] +"\n"+ item['content']

                  if 'nextChunks' in add_chunks:
                      n_chunks = chunk['chunkMetadata']['nextChunks']
                      if n_chunks:
                          for n_chunk in n_chunks:
                              item['content'] = item['content'] +"\n"+ n_chunk['content']

              search_results['result'] = item

      return search_results

  #----------------------------------------------------------------------------------------------------------------
  def ranking_results(self, query, search_results, top_n):

      records = []

      for index, response in enumerate(search_results):

        title = response['result']['title']
        content = response['result']['content']
        records.append(discoveryengine.RankingRecord(id=str(index), title=title, content=content))

      request = discoveryengine.RankRequest(
          ranking_config = Controller.ranking_config,
          model = env.ranker_model,
          top_n = top_n,
          query = query,
          records = records
        )

      ranked_response = Controller.re_rank_client.rank(request=request,)

      ranked_res_list = []

      for record in ranked_response.records:  # https://cloud.google.com/generative-ai-app-builder/docs/reference/rpc/google.cloud.discoveryengine.v1alpha#rankresponse

        if record.score > env.rank_score:
          print(f"ranking score > {env.rank_score} : {record.score}")
          print(f"Ranked result : [{record.score}] : {record.content}")
          ranked_res_list.append(record.content)
        else:
          print(f"ranking score < {env.rank_score} : {record.score}")
      return ranked_res_list


  #----------------------------------------------------------------------------------------------------------------

  def gemini_response(self, prompt:str,
                  response_schema:dict = None):

      model = GenerativeModel(model_name= env.model)

      generation_config = GenerationConfig(
          temperature=0.5,
          top_p=1.0,
          top_k=32,
          candidate_count=1,
          max_output_tokens=8192,
          )

      responses = model.generate_content(
          [prompt],
          generation_config = generation_config)

      print(f"[Controller][call_gemini] Final response Len {len(responses.text)}")

      return responses.text
