from datetime import datetime
from typing import List, Dict
import asyncio
import os
from together import AsyncTogether, Together
from itertools import cycle
import json
from openai import AsyncOpenAI
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
import time as time_module

aggregator_model ="gemini-2.5-pro-preview-05-06"# "Qwen/Qwen2.5-72B-Instruct-Turbo"
reference_models = [
# "meta-llama/Llama-3.3-70B-Instruct-Turbo",
# "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
# "Qwen/Qwen2.5-7B-Instruct-Turbo",
# "mistralai/Mixtral-8x7B-Instruct-v0.1",
# "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
# "deepseek-reasoner",
# "o3-mini"

  # "gemini-2.0-flash-lite",
  # "gemini-2.0-flash",
  # "gemini-2.5-pro-preview-05-06"
"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
"meta-llama/Llama-4-Scout-17B-16E-Instruct"
]

async def llm_deepseek(system_prompt: str, user_prompt: str) -> str:
      client = AsyncOpenAI(api_key="sk-665ff0ad09c041f3be013b8ac1d9cb82", base_url="https://api.deepseek.com")
      response = await client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt},  # Uncomment and fix role to "system"
            {"role": "user", "content": f"{user_prompt}."},  # Explicit key names
        ],
      )
      return response.choices[0].message.content

async def llm_gemini(system_prompt: str, user_prompt: str, model:str) -> str:
      client = genai.Client(api_key="AIzaSyBihb4YNNdT7M5M5MMorrqvjmMX9Q-GD_Y",http_options=HttpOptions(api_version="v1"))
      # nếu gemini bản 2.5 thì sài
      response = await client.aio.models.generate_content(
        model=model,
        contents=f'System: {system_prompt}\nUser: {user_prompt}",',
        config=GenerateContentConfig(
            temperature=0.1),
      )
      return response.text

async def llm_together(system_prompt: str, user_prompt: str, model: str) -> str:
    TOGETHER_API_KEY = '9902419d89b4b07e34287de0630031ac014692ddfadf9102a2f4d9e66cdf38d7'
    # client = Together(api_key=TOGETHER_API_KEY)
    async_client = AsyncTogether(api_key=TOGETHER_API_KEY)
    response = await async_client.chat.completions.create(
                  model=model,
                  messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{user_prompt}."},],
                  temperature=0.7,
                  max_tokens=4096,
              )
    return response.choices[0].message.content

async def llm_together1(system_prompt: str, user_prompt: str, model: str) -> str:
    async_client = AsyncTogether(api_key="1d639cc791c5f420c1c7ea1ed387383d0b037be515ad5bab67d32080cf7bcdcc")
    response = await async_client.chat.completions.create(
                  model=model,
                  messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{user_prompt}."},],
                  temperature=0.7,
                  max_tokens=4096,
              )
    return response.choices[0].message.content

async def llm_together2(system_prompt: str, user_prompt: str, model: str) -> str:
    async_client = AsyncTogether(api_key="1eab93d3f2f8f91303711c7824335ea7ec0936f60fb7927b376d55966f153289")
    response = await async_client.chat.completions.create(
                  model=model,
                  messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{user_prompt}."},],
                  temperature=0.7,
                  max_tokens=4096,
              )
    return response.choices[0].message.content


async def run_llm(model: str,system_prompt:str, prompt: str,count: int) -> str:
    """Run a single LLM call."""
    for sleep_time in [1, 2, 4]:
      try:
          # response = await llm_gemini( system_prompt, prompt, model)
          # if model == "gemini-2.0-flash":
          #     response = await llm_gemini( system_prompt, prompt)
          # elif model == "gpt-4o":
          #     response = await llm_gpt(system_prompt, prompt)
          # elif model == "deepseek-reasoner":
          #     response = await llm_deepseek(system_prompt, prompt)
          # else:
          print(f"Running model {model} with prompt: {prompt}, count: {count}")
          response = ''
          if count % 3 == 0:
            response = await llm_together2(system_prompt, prompt,model)
          elif count % 3 == 1:
            response = await llm_together1(system_prompt, prompt,model)
          else:
            response = await llm_together(system_prompt, prompt,model)
          if isinstance(response, str):
            response = clean_json_string(response)

          return response
      except Exception as e:
        await asyncio.sleep(sleep_time)
        return f"Error running model {e}"
    return f"Error running model {model}"

async def method(system_prompt: str, user_prompt: str, aggregator_prompt: str ) -> str:
    results = await asyncio.gather(*[run_llm(model, system_prompt, user_prompt, i) for i, model in enumerate(reference_models)])
    # print(f"Results: {results}")
    # write_to_json(results, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_results.json')
    aggregator_system_prompt =  """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Format your response like provided response in JSON format. Do not include any additional text or comments."""

    if aggregator_prompt == "":
        aggregator_prompt = aggregator_system_prompt

    # finalStream = client.chat.completions.create(
    #     model=aggregator_model,
    #     messages=[
    #         {"role": "system", "content": aggregator_prompt + "\n" + "\n".join([f"{index+1}. {str(element)}" for i, element in enumerate(results)])},
    #         {"role": "user", "content": user_prompt},
    #     ],
    #     stream=True,
    # )
    aggregator_system_prompt =  """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Format your response like provided response in JSON format. Do not include any additional text or comments."""

    client = genai.Client(api_key="AIzaSyBihb4YNNdT7M5M5MMorrqvjmMX9Q-GD_Y")

    response =  client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents= user_prompt,
        config= GenerateContentConfig(
            system_instruction= aggregator_system_prompt + "\n" "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)]),
            # max_output_tokens=3,
            # temperature=0.3,
        ),
    )
    # print(response.text)
    # print(response.text)
    # response = ""
    # for chunk in finalStream:
    #     response += chunk.choices[0].delta.content or ""
    return response.text

async def method_aggregator(list: List, system_prompt: str,  user_prompt: str, aggregator_prompt: str) -> str:
    tasks=[]
    count = 0
    for model, item in zip(cycle(reference_models), list):
        user_prompt = f'{user_prompt}\nThe phase you are working on is {item}',
        task = run_llm(model,system_prompt, user_prompt, count)
        count += 1
        tasks.append(task)
    results = await asyncio.gather(*tasks)

    aggregator_prompt=aggregator_prompt + "\n" + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])

    client = AsyncOpenAI(api_key="sk-665ff0ad09c041f3be013b8ac1d9cb82")
    response = await client.chat.completions.create(
    model="o3-mini",
    messages=[
        {"role": "system", "content": system_prompt},  # Uncomment and fix role to "system"
        {"role": "user", "content": f"{user_prompt}."},  # Explicit key names
    ],
    )
    return response.choices[0].message.content

import re
import json
def clean_json_string(json_string):
    if json_string.startswith("```json") and json_string.endswith("```"):
        # Loại bỏ các dấu backtick và từ khóa json
        json_content = json_string[7:-3].strip()
        try:
            # Phân tích JSON
            data = json.loads(json_content)
            return data
        except json.JSONDecodeError:
            print("Invalid JSON content")
            return json_string
    else:
        print("Content is not valid Markdown JSON")
        return json_string

def write_to_json( data, filename: str) -> None:
    """Write data to a JSON file."""
    with open(filename, 'w') as f:
      json.dump(data, f, ensure_ascii=False, indent=4)

requirements_list = ["Study Social Network (SSN) is a social network for students and researchers to share studying materials, findings, and experiences in learning, teaching and research. Through SSN you are able to share your study experience in both art and science. With SSN, students can find useful learning information about Algebra A1, Method of Report Writing, and other subjects of study. Students can evaluate and rate teachers and courses via this network. Students can also provide materials and recommendations for each course. This information helps other students to select suitable courses that match their interests, studying and career goals. For researchers and teachers, they can share teaching and research experiences. This network is also a place for them to find potential students to work with. The initial focus would be on the universities in Vietnam. Each university is a network, and thus the materials and experiences are university-based. Of course, one person can join multiple networks if they are interested in. In the future, the system is expanded to the network of high schools and elementary schools in Vietnam.","Ban Nha Nong (BNN) ltd. is a software development company specializing in developing and maintaining software products and providing software services that promote the application of information technology in rural regions in Vietnam. The market for the company is enormous as the large majority of population in Vietnam is farmers. Now, BNN wants to develop and deploy a system called www.hailua.com.vn. The website is intended to be a place for farmers to sell their produces (rice, fruits, fish, etc.) and for customers to buy farmers' produces. It is an e-commerce website where any farmer can register an account given they have a bank account and can post their produces with descriptions, images, prices and any other things related to the produces. They can collect the money after their produces are sold. Customers can go to the website, searching for the produce they want, selecting the produce, paying online, and shipping it to their house. They can also compare the prices and features of different produces. Generally, the website is something similar to amazon.com and ebay.com. The company plans to deploy the system in multiple periods. The initial version does not necessarily include all fancy features like those of amazon.com or ebay.com. But it should have basic capabilities allowing posting, selling, and buying produces online. Shipping and payment should be done online as well",
    "Cát Đằng Coffee (CDC) is one of the ideal places for people to hang out during their free time_module. There are different distinct areas that are designed to suit a variety of purposes. Among them, the luxurious section is suitable for friend meetings and business discussions. The romantic garden court is a remarkable choice for loving couples. CDC is a popular trademark of Én Bạc Co., Ltd. CDC has become a big system with many branches. Therefore, they have a need of building a system to manage their business. CDC has many branches across the country. The goal of the system is to combine and manage the business information from all of the branches, and toprovide support for staff at each local branch. When customers call drinks, waiters take order and transfer the order to the preparation room. The system should allow waiters to perform their tasks easily and minimize their movements. The system should also produce bills automatically. Moreover, the system allows the company to generate and view revenue reports of each local branch and the whole chain.",
    "Cơm Tấm Cali (CTC) is restaurants specialized on broken rice in Ho Chi Minh City. CTC currently has over 10 branches with many kinds of broken rice and special drinks in a luxurious space. The serving style is professional nd elegant.CTC wants to develop and deploy a system to manage branches and allow their customers to order via the Internet. Customers are easily able to choose and place food orders on the website which automatically transfers the customer request to the nearest branch. CTC usually has promotions, and customers should be provided the way to keep track and apply them. The website allows customers to pay the bill online by www.nganluong.vn or another system like that. Besides, customers are also able to give feedbackthrough the website, which shows that the website is willing to listen to customer's feedback and improve the website on any feedback given. Customers can also rate the restaurant besides their comments. The prices of the dishes are updated regularly and correctly, making sure that the website is up to date. This feature makes the customers feel safe and secure when surfing the website. The website would provide a new service ordering broken rice online that is in high demands by office workers currently. The system should be able to manage all branches so that data analysis can be done to support business decisions and strategies.",
    "Dream Social Network (DSN) is an online community where members can post their ideas, talk about their dreams, or share moods, individual opinions, etc. When joining the DSN, everyone has a particular space to expose all of their thinking and feelings such as sadness, happiness, hopelessness, and silliness. The idea behind this network is that, sometimes, we have many ideas but we don't know how to share them with our friends, how to survey or how to get feedback. DSN would help us to share our ideas with everyone around the world. You want to share your sweet dreams and find someone having the similar dreams? DSN is a good place for that. Maybe you can find someone who has you in his/her dreams. It's good to get connected in such case, and you and that person might become soul mates. You want to share your ideas and find solutions to achieve your ideas? In this network, everyone can see your ideas and dreams. DSN is a good place to remind your dream and to make it come true.",
    "Hoàng Sĩ Pig Farm is a big farm that supplies different kinds of piglets (young pigs) to farmers. Currently, the farm has over 200 piglets. The farm owner wants to develop a system to manage and monitor the growth, feeding, temperature and environmental criteria for each kind of piglet. The system also helps farmers easily keep information and exchange piglets in outlying areas. The website has two groups of users. The first one is the farm's staffs. They should be able to update the information for growth, health and common diseases for each kind of pigs, import and export states, and revenue statistics. The second user group is farmers who are the farm's customers. They can register an account on the system to look over all kinds of pig supplied by the farm, and order one or many piglets needed. After buying the piglet, the farmer uses the system to communicate with the farm to make sure that the piglet is properly raised. Feeding and vaccination procedures, for example, are also managed and communicated via the system. The farm owner can easily manage, keep the information up-to-date, and announce promotions. The website is useful for owner to make monthly, quarterly, and annual reports.",
    "The ITT website is a comprehensive platform offering detailed information on various travel tours and related articles. It introduces an overview of the company's services, showcases featured tours, and highlights the latest news and special promotions. The tour system is organized into key categories including Special Tours, Japan Tours, Asia Tours, and Europe-Australia-America Tours. Additionally, the website provides visa consultation and support services for multiple countries. Cultural articles and travel experience stories enrich the content, helping travelers gain deeper insights into destinations. Customer sharing sections feature real feedback, stories, and reviews from past clients, enhancing the site's credibility. The website offers multiple contact options, including an online inquiry form, dedicated hotlines for individual travelers, group tours, and visa services, alongside email and office address information. Hotline numbers are constantly displayed to ensure fast and easy support. Each tour detail page includes a comprehensive day-by-day itinerary, highlights of the trip, service information (such as hotel accommodation, flight tickets, and meals), detailed tour pricing, terms of included and excluded costs, illustrative images, guide information, and clear policies on payment methods and cancellation conditions",
    ]
# vì requirements là mảng nên là tối muốn chạy tất cả item của requiremnt chạy tất cả flow từ trên xuống dưới, viết giúp tôi đoạn code đó, đồng thời ghi vào file kết quả từng giai đoạn endTime- start_time ghi cả về thời gian và kết quả response

async def run():
  for index in range(len(requirements_list)):
    execution_times = []
    print(f"\n{'='*50}")
    print(f"Processing requirement {index+1}/{len(requirements_list)}")
    print(f"{'='*50}")
    try:
      os.mkdir(f"{index+1}")
    except:
      print(f"Directory {index+1} already exists.")
    
    try: 
      requirements = requirements_list[index]
      print(f"Processing requirement {index+1}: {requirements}")
      start_time = time_module.time()
      summary_prompt = {
        "system_prompt" : '''You are a paraphrase specialist.Your task is to paraphrase the project description described in the provided user prompt. Output result in a paragraph in English with no format needed.
        Ensure output format:
        {{
          "sumarize_req": "summary of the project"
        }}
        NO additional keys, comments or text.
        ''',
        "user_prompt" : f'''The project description is {requirements}
        '''
      }
      summarize_req = await method(summary_prompt['system_prompt'], summary_prompt['user_prompt'], '')

      feature_prompt = {
        "system_prompt" : '''You are a feature generation specialist. Your task is to generate a list of features based on the requirements provided.
        Ensure output format is list of dict, format of dict is:
        [{{"category": "category 1",
              "list_feature": [""]}},
              {{"category": "category 2",
              "list_feature": [""]}},
              {{"category": "category 3",
              "list_feature": [""]}},
              ...
          ]
        NO additional keys, comments or text.
        ''',
        "user_prompt" : f'''Decide all the features for the given project description described in the square bracket. Only list features according to the description. Each object include feature category and list of features. The requirement is {summarize_req}
        '''
      }
      feature_list = await method(feature_prompt['system_prompt'], feature_prompt['user_prompt'], '')
      end_time = time_module.time()
      execution_times.append(end_time - start_time)
      print(f"Time taken: {end_time -start_time } seconds")
      summarize_req = clean_json_string(summarize_req)
      write_to_json(summarize_req, f"{index+1}/summarize_req.json")

      feature_list = clean_json_string(feature_list)
      print(type(feature_list))  # <class 'list'>
      print(feature_list)     # dict đầu tiên
      write_to_json(feature_list, f"{index+1}/feature_list.json")

      # feature flow
      start_time = time_module.time()
      feature_flow_prompt = {
          "system_prompt":'''You are a professional product owner. When returning a feature to the user at the final step, ensure each returned feature has corresponding flows and screens.
          For each feature in the given category, your task is to clearly outline the complete user flow and list all the screens required for that feature. Use the provided project description as context. If any explicit data is missing, creatively generate a logical and detailed flow based on common practices for similar features. The flow should include step-by-step actions along with the user roles involved. Additionally, list each required screen with a short description of its purpose.

        Ensure that the final output is in JSON format matching the following structure:
        {{
          "category": "<Category Name>",
          "features": [
            {{
              "featureName": "<Feature Name>",
              "flow": "<Detailed step-by-step flow with user roles>",
              "screens": [
                "<Screen 1: Description>",
                "<Screen 2: Description>",
                "..."
              ]
            }}
          ]
        }}
      NO additional keys, comments or text
        If any information is missing, please generate a reasonable flow and screen list. The output JSON must not contain any fields with null values; instead, provide placeholder steps or notes indicating further details are needed.''',
          "user_prompt" : f'''The project description is ${summarize_req} and the feature list is ${feature_list}. '''
      }
      aggregator_prompt = """You have been provided with list of each category from various open-source models. Your task is to synthesize these into a single, high-quality JSON that consolidates all phases. Evaluate and merge the information carefully and ensure the final JSON is coherent and accurate.
      output format for each task:
        {{
          "category": "<Category Name>",
          "features": [
            {{
              "featureName": "<Feature Name>",
              "flow": "<Detailed step-by-step flow with user roles>",
              "screens": [
                "<Screen 1: Description>",
                "<Screen 2: Description>",
                "..."
              ]
            }}
          ]
        }}
      NO additional keys, comments or text.\n\n"""

      tasks = []
      print("Reference models:", reference_models)
      print("Feature list:", feature_list)

      results=[]
      if isinstance(feature_list, list):
        count = 0
        for model, item in zip(cycle(reference_models), feature_list):
            try:
                print("Processing category:", item)
                user_prompt = feature_flow_prompt['user_prompt'] + f'\nThe category you are working on is {item}'
                task = run_llm(model, feature_flow_prompt['system_prompt'], user_prompt,count)
                count += 1
                tasks.append(task)
            except Exception as e:
                print(f"Error creating task for category {item}: {e}")

        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error during gather: {e}")
            print("Tasks that failed:", tasks)

        # check result bị lỗi thì retry
        retry_tasks = []
        for idx, result in enumerate(results):
            if isinstance(result, str) and "Error running model" in result:
                print(f"Retrying task {idx} due to error: {result}")
                model = reference_models[(idx + 1) % len(reference_models)]
                category = feature_list[idx]
                user_prompt = feature_flow_prompt['user_prompt'] + f'\nThe category you are working on is {category}'
                retry_tasks.append(run_llm(model, feature_flow_prompt['system_prompt'], user_prompt,idx+1))

        if retry_tasks:
            print("Retrying failed tasks...")
            retry_results = await asyncio.gather(*retry_tasks)
            for i, result in enumerate(results):
                  if isinstance(result, str) and "Error running model" in result:
                    results[i] = retry_results.pop(0)

      end_time = time_module.time()
      print(f"Time taken: {end_time - start_time} seconds")
      execution_times.append(end_time - start_time)
      write_to_json(results, f"{index+1}/feature_flow.json")

      roadmap_prompt = {
            "system_prompt" : '''You are a A talented project manager. Your task is create a project roadmap based on the project description and feature list. The results must include the following keys: id, phase, feature list, estimatedTime, startTime, endTime, and dependencies. The id should follow the format 'Px' where x is a sequential number starting from 1. The feature list should be a list of features chosen from the given feature list (in square brackets). Consider that phases which can be executed in parallel are allowed. Assume that this project starts today.
            Important: Return ONE JSON object list. Output NOTHING except the JSON object list.
            Format of object in list
              json```
                  {
                  "id":
                  "phase":
                  "feature_list" : [],
                  "estimatedTime": ,
                  "startTime": ,
                  "endTime": ,
                  "dependencies": []
                }
            ''',
            "user_prompt" : f'''The project description is ${summarize_req} and the feature list is ${feature_list}. '''
          }
      start_time = time_module.time()
      roadmap = await method(roadmap_prompt['system_prompt'], roadmap_prompt['user_prompt'],'')
      end_time = time_module.time()
      # lưu thời gian vào file time_module.txt
      execution_times.append(end_time - start_time)
      print(f"Time taken: {end_time - start_time} seconds")

      roadmap = clean_json_string(roadmap)
      print(type(roadmap))  # <class 'list'>
      print(roadmap)      # dict đầu tiên
      write_to_json(roadmap,f"{index+1}/roadmap.json")
      print("Waiting 30 seconds before next phase...")
      await asyncio.sleep(30)

      task_prompt = {
            "system_prompt" : '''You are A professional project manager and a product owner. When returning tasks to the user at the final step, ensure that all tasks are included. Create a list of tasks based on the given project description, project roadmap, and feature list. Make sure to include all necessary tasks and subtasks if needed. Each phase should include tasks for design, implementation, and testing. Each task must include an id (format: Px-Ty, where x and y are sequential numbers starting from 1 for the phase and the task respectively), taskName, correspondingPhase, correspondingFeature, priority (choose only from High, Medium, or Low), estimationTime (calculated in hours, days, or weeks), startTime, endTime, dependencies, and responsiblePerson. Ensure that tasks cannot be broken down into smaller tasks and that the estimation time is as short as possible while still being sufficient for skilled employees to complete the work. Dependencies should be an array of task ids, if applicable. Output an array in dictionary format.
            Ensure that the final output is in JSON format matching the following structure:
                  {{
                    "id":"<Task ID>",
                    "taskName": "<Task Name>",
                    "correspondingPhase": "<Corresponding Phase>",
                    "correspondingFeature": "<Corresponding Feature>",
                    "priority": "<High, Medium, Low>",
                    "estimationTime": "<Estimated Time>",
                    "startTime": "<Start Time>",
                    "endTime": "<End Time>",
                    "dependencies": ["<Dependency 1>", "<Dependency 2>"],
                    "responsiblePerson": "<Responsible Person>"
                  }}
            NO additional keys, comments or text.''',
            "user_prompt" : f'''The project description is ${summarize_req}, the project roadmap is ${roadmap}, and the feature list is ${feature_list}.',
            '''
          }
      aggregator_prompt = """You have been provided with list of each phase from various open-source models. Your task is to synthesize these into a single, high-quality JSON that consolidates all phases. Evaluate and merge the information carefully and ensure the final JSON is coherent and accurate.
      output format for each :
        {{
        "id":"<Task ID>",
        "taskName": "<Task Name>",
        "correspondingPhase": "<Corresponding Phase>",
        "correspondingFeature": "<Corresponding Feature>",
        "priority": "<High, Medium, Low>",
        "estimationTime": "<Estimated Time>",
        "startTime": "<Start Time>",
        "endTime": "<End Time>",
        "dependencies": ["<Dependency 1>", "<Dependency 2>"],
        "responsiblePerson": "<Responsible Person>"
        }}
      NO additional keys, comments or text.\n\n"""

      tasks=[]
      print(reference_models)
      start_time = time_module.time()
      if isinstance(roadmap, list):
        count = 0
        for model, item in zip(cycle(reference_models), roadmap):
            print("Phase: ",item)
            user_prompt = task_prompt['user_prompt'] + f'\nThe phase you are working on is {item}',
            task = run_llm(model,task_prompt['system_prompt'], user_prompt,count)
            count += 1
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        # check result bị lỗi thì retry
        retry_tasks = []
        for i, result in enumerate(results):
            if isinstance(result, str) and "Error running model" in result:
                print(f"Retrying task {i} due to error: {result}")
                model = reference_models[(i + 1) % len(reference_models)]
                phase = roadmap[i]
                user_prompt = task_prompt['user_prompt'] + f'\nThe phase you are working on is {phase}'
                retry_tasks.append(run_llm(model, task_prompt['system_prompt'], user_prompt,i+1))

        if retry_tasks:
            print("Retrying failed tasks...")
            retry_results = await asyncio.gather(*retry_tasks)
            for i, result in enumerate(results):
                if isinstance(result, str) and "Error running model" in result:
                    results[i] = retry_results.pop(0)

      end_time = time_module.time()
      print(f"Time taken: {end_time - start_time} seconds")
      execution_times.append(end_time - start_time)
      # Write final results to file
      write_to_json(results, f"{index+1}/task_list.json")

      print(f"Time taken: {end_time - start_time} seconds")

      # write_to_json(execution_times, f"{index+1}/time_module.txt")
      with open(f"{index+1}/execution_times.txt", 'w') as f:
          for elapsed_time in execution_times:
              f.write(f"{elapsed_time}\n")
    except Exception as e:
      print(f"Error processing requirement {index+1}: {e}")
      continue
    print(f"Waiting 1 minute before processing the next requirement...")
    await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(run())
    # run()
    # for i in range(len(requirements_list)):
    #   try:
    #     os.mkdir(f"{index+1}")
    #   except:
    #     print(f"Directory {index+1} already exists.")