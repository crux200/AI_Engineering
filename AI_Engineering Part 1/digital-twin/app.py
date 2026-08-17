import os
from openai import OpenAI 
import gradio as gr
from pprint import pprint
import chromadb
import uuid
import spaces
import torch
import json
import requests
import random

# 1. Dummy function to satisfy Hugging Face ZeroGPU startup check
@spaces.GPU
def initialize_gpu_space():
    # Simple tensor operation to activate ZeroGPU system
    _ = torch.zeros(1).cuda()

# Call the initialization right away
initialize_gpu_space()

###-----
# Setup
#------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API key is missing")
client = OpenAI()

###-----
# Documents
#------

document_overview= """

VT is a Technical Program Manager with a passion for AI and machine learning. He has experience in developing web applications, data analysis, and cloud computing. VT enjoys solving complex problems and is always eager to learn new technologies. In his free time, he likes to read about the latest advancements in AI.

Other Career History:
2004- 2020: Worked for multiple companies in the IT sector and mainly on Microsoft Projects and Technologies. He has experience leading teams and managin projects in fact-pacing environment
2020- present: VT has been working as Technical Program Manager at Princess cruises as a contractor. Most recently, he built the OceanSafety ERP Platform that removed the paper trail across the fleet for managing the emergency instructions and procedures. 

What drives him: He loves to play cricket and  is a die-hard fan of the sport. He is also passionate about mentoring and helping others grow in their careers. VT believes in continuous learning and is always looking for opportunities to improve his skills and knowledge.

His approach: VT approaches challenges with a systematic and analytical mindset. He believes in breaking down complex problems into manageable components and leveraging his technical expertise to find innovative solutions. He is committed to delivering high-quality results while fostering a collaborative environment that encourages continuous improvement and knowledge sharing.

Communication Style: VT is known for his clear and concise communication style. He is able to convey complex technical concepts in a way that is easily understandable to both technical and non-technical stakeholders. He values open and transparent communication and encourages feedback from team members to ensure alignment and clarity.

Additional Info:
- In 2001, VT graduated with a Bachelor's degree in Computer Applications from Osmania University. He gained experience in software development through various projects. One of the projects he worked on was neural networks in VC++ language.",
- VT is an experimental cook and enjoys trying out new recipes and techniques in Kitchen. The first dish he learned to cook was a Kichdi, a traditional Indian dish made with rice and lentils. He learned this recipe from online videos and cooking became a necessity for him when he moved to the US and had to cook for himself. He enjoys experimenting with different cuisines and flavors, and often incorporates his own twists to traditional recipes. Cooking has become a creative outlet for him, allowing him to unwind and express himself in the kitchen.",
- VT is a die-hard cricket fan and enjoys playing the sport in his free time. He has played at different levels and teams and represented the Microsoft Cricket Club in  NWCL and was part of the championship team in 2011. He also has scored a century in one of the games in the same year. He is a bowling all rounder and out swing is his specialty. He has also played in ARCL and taken 100+ wickets.",
- VT is a biryani lover and enjoys cooking and eating this popular Indian dish. He has tried different variations of biryani, including Hyderabadi, Tamil and Kerala styles. He also likes to experiment with his own recipes and has created his own version of biryani that he enjoys making for family and friends. Biryani is one of his favorite comfort foods, and he often seeks out the best biryani restaurants wherever he travels.",
- VT has experience in managing Windows Update deployments especiall the Windows device driver ecosystem and has worked on projects related to Windows Update driver servicing for multiple companies. He has knowledge of the Windows Update process, including the different types of updates, deployment methods, and troubleshooting techniques. He has also worked on automating the Windows Update process using scripts and tools to improve efficiency and reduce downtime. His expertise in this area has helped IHV and OEM Partners ensure that their systems are up-to-date and secure. He has also authored several INFs that are on WU catalog even today."

"""
document_education="""
University of Hyderabad logo
University of Hyderabad

Post Graduate Diploma in Project Management, Project Management

2015 – 2016

The course was designed to provide the most refined skills in understanding economic aspects of planning coupled with managerial aspects. It addresses the professional needs of managers, executives and industrialists engaged in planning, management and execution of projects of varied nature by providing them with the necessary theoretical orientation and also discussion and analysis of practical problems in project appraisal/feasibility, apart from management of finance, marketing and human resources.

The project work "uTrash- The complete waste management service" was also accepted by the university. 

Vinayaka Mission's Research Foundation - University logo
Vinayaka Mission's Research Foundation - University

MCA, Computer Applications

2003 – 2006

Osmania University logo
Osmania University

BCA, Computer Applications

2000 – 2003"""
document_professional_experience=""""

Hughes Systique Corporation (HSC) logo
Senior Principal Engineer

Hughes Systique Corporation (HSC) · Full-time

Nov 2023 - Jun 2026 · 2 yrs 8 mos

OceanSafety Platform — Mission-Critical SaaS
•Lead and manage a team of software engineers delivering a mission-critical, Ocean Safety platform across PCL SHIP environments — owning team performance, professional development, sprint planning, and delivery accountability end-to-end.
• Drive product roadmap execution in close collaboration with Product Owners and business stakeholders — translating strategic objectives into engineering backlog priorities and ensuring on-time, high-quality feature delivery.
• Architect and govern a Java-based distributed system (microservices, REST APIs) underpinned by Couchbase and Oracle databases — maintaining high availability and fault tolerance across a globally deployed, safety-critical platform.
• Serve as subject matter expert and senior technology advisor to business stakeholders — communicating roadmap status, architectural decisions, and risk posture at the executive level.
•Consistently exceed customer satisfaction targets through delivery of impactful enhancements and automation that measurably reduce operational overhead.… more

 Technical Project Leadership and Technology Management

Launch Consulting Group logo
Launch Consulting Group

5 yrs 4 mos

Technical Program Manager

Jan 2021 - Oct 2023 · 2 yrs 10 mos

Managed end-to-end release planning and deployment for OceanSafety and Brand Experience products — consumer and crew-facing mobile and web applications serving passengers and crew across Princess Cruises' global fleet.… more

Technical Project Manager

Full-time

Jul 2018 - Dec 2020 · 2 yrs 6 mos

Bellevue, Washington · Hybrid

Responsible for managing the Azure Devops, Power Platform, Cloud Strategy for the Business Unit.

•	Conduct and participate in release planning for the team(s).
•	Motivate, integrate, and build relationships with cross-functional team members, stakeholders
•	Increased operational efficiency by more than 50% 
•	Experience leading projects involving Data Factory, Data Lake, PowerBI and other cloud technologies. 
•	Managed the Azure AD & migration efforts for the Partner Center team(s).

… more

Inspur Worldwide Services logo
Tech PM

Inspur Worldwide Services · Full-time

Jul 2016 - Jun 2018 · 2 yrs

Bellevue

Developed and improved the capabilities of Driver Publishing team(s).
•	Collaborated with team(s) to quickly resolve driver discrepancies.
•	Developed processes and managed stakeholder communication(s).
•	Facilitated stakeholder meetings and developed PowerShell scripts to improve device driver quality.
•	Content authoring and continuous improvements resulted in good quality drivers in the Windows Ecosystem. 
… more

Program Manager

Robustware INC · Full-time

Jul 2013 - Jun 2016 · 3 yrs

Redmond

Planning of resources, schedules for the publishing teams(s). 
•	Program delivery and spec review for Publishing & Model Based Servicing Projects.
•	Increased Customer satisfaction by over 20% for Driver not found (DNF) responses.
•	Involved in driver servicing for system-on-chip (SoC) partners. 
•	Conducted stakeholder reviews for device driver removals from Windows Update.
… more

Wipro Limited logo
Solution Delivery Analyst

Wipro Limited · Full-time

Jan 2008 - Jun 2013 · 5 yrs 6 mos

Redmond, Washington

Worked with external and internal partners, 
Driver content authoring & WU detection Logic for in-house projects.
Debug/triage WU driver install issues
Mark drivers as Important on WU.

Managed the Windows Logo Certification program.
•	Provided email and web support for Windows Hardware Online Services (WHQL) Partners. 
•	Participated in Hardware Logo Events
•	Developed processes and tooling for minimizing the team efforts.
•	Achieved CSAT above 7 for the Logo certification program initiatives."""

###-----
# Chunking Function
#------
BOUNDARIES = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split `text` into overlapping chunks of up to `chunk_size` characters,
    each overlapping the previous chunk by `overlap` characters, cutting at
    the nearest natural boundary (paragraph break, newline, sentence end,
    comma, then space) past the halfway point of the chunk.
    """
    chunks = []
    n = len(text)
    half = chunk_size // 2
    start = 0
    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            chunk = text[start:end]

            for boundary in BOUNDARIES:
                pos = chunk.rfind(boundary)
                if pos != -1 and pos >= half:
                    end = start + pos + len(boundary)
                    break

        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap

    return chunks

###-----
# RAG: Chunk, Embed and Store in ChromaDB
#------

#Chunk
documents=[
    {"text":document_overview, "source":"Overview"},
    {"text":document_education, "source":"Education"},
    {"text":document_professional_experience, "source":"Professional experience"}
]

chunks=[]
ids=[]
metadatas=[]

for doc in documents:
    #prepare the lists
    chunks_ =chunk_text(doc["text"], chunk_size=300, overlap=30)
    ids_ =[str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ =[{"source":doc["source"], "chunk_index":i} for i in range(len(chunks_))]

    #Add to main lists
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

#Print for logs
print (f"Created {len(chunks)} chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (ID: {ids[i]}, Source: {metadatas[i]['source']}, Index: {metadatas[i]['chunk_index']}, Length: {len(chunk)}")
    print(chunk)
    print()

#Generate Embeddings
#client=OpenAI()
response= client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks
)
embeddings= [item.embedding for item in response.data]

#Verify Embeddings for logs
print(f"Generated  {len(embeddings)} embeddings:\n")
print(f"Each embedding has {len(embeddings[0])} dimensions\n")

#Initialize ChromaDb and Store Vectors
#initialize chromadb client(Persitent storage)
chroma_client= chromadb.PersistentClient(path="./chroma_db_twin")

collection = chroma_client.get_or_create_collection(name="DigitalTwin")

#Get or Create + Empty the collection before adding new data(
if collection.get()["ids"]:
   collection.delete(collection.get()["ids"])

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)

pprint(collection.get())

###-----
# Tools
#------
tools=[]
#Add tool calling functionality(PushOver)
#Set up Pushover
pushover_user=os.getenv("PUSHOVER_USER")
pushover_token=os.getenv("PUSHOVER_TOKEN")
pushover_url="https://api.pushover.net/1/messages.json"


#Create send_Notification function
def send_notification(message:str):
    if pushover_user is None or pushover_token is None:
        return "Notification failed: Pushover not configured"
    payload={"user":pushover_user,"token":pushover_token,"message":message}
    requests.post(pushover_url,data=payload)
    return f"Notification sent:{message}"
#Step3: Describe Pushover as an LLM tool
send_notification_function={
    "name": "send_notification",
    "description": "Sends a push notification to the real Varghese. Use this when: 1) Someones wants to get in touch, hire or collaborate- ask their name and contact details first, then send notification to Varghese with the name and contact details. 2) You dont know the answer to a question about varghese- Send Automatically without asking, including the question so he can add the info later.",
    "parameters": {
        "type":"object",
        "properties":{
            "message":{
                "type":"string",
                "description":"The notification message to send to the user's device"
            }
        },
        "required":["message"]
    }
    
    }
#Add Pushover to the list of tools
tools.append({"type":"function","function":send_notification_function})

#simulates rolling a six-side die

def dice_roll():
    result = random.randint(1,6)
    return result
#Describe function to the LLM 
roll_dice_function={
"name": "dice_roll",
    "description": "Simulates rolling a single six-side dice. Use this to get a random dice roll.",
    "parameters": {
        "type":"object",
        "properties":{},
        "required":[]
    }

}

#Add function to the list of LLM
tools.append({"type":"function","function":roll_dice_function})

###-----
# Tool: Handler
#------

#Function to handle LLM tool calls 
def handle_tool_call(tool_calls):
    tool_results=[]
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args= json.loads(tool_call.function.arguments)

        if function_name=="send_notification":
            content= send_notification(args["message"])
        elif function_name == "dice_roll":
             content=f"Rolled {dice_roll()}"
        #elif function_name="Insert_function-3":
           # content=Insert_function-3{args["message"]}"

        else:
            content = f"unknown function: {function_name}"

   # print(f"sent notification: {args['message']}")
        tool_call_result={
            "role":"tool",
            "content": content,
            "tool_call_id":tool_call.id
        }
        tool_results.append(tool_call_result)
    return tool_results
###-----
# System Message
#------
system_message = """You are the digital twin of VT. You are a helpful assistant. When people talk to you, you respond as if you are VT — in first person, using his voice, personality, and knowledge.

SCOPE:
- Only answer questions that are about VT, his experiences, opinions, or knowledge as provided to you.
- If a question is unrelated to VT (general knowledge, other topics, requests to perform unrelated tasks, etc.), politely decline and redirect the conversation back to VT-related topics. Do not answer it, even partially.

FACTUAL ACCURACY:
- Only use the factual information about VT that has been explicitly provided to you. Do not invent, infer, guess, or embellish any facts about VT.
- Do not use your own general/pretrained knowledge to fill in gaps about VT — even plausible-sounding details. If it wasn't given to you as a fact about VT, treat it as unknown.
- Do not search the internet or use any external source to find or "verify" facts about VT. Your knowledge of VT is limited strictly to what's in your provided context.
- If you do not know the answer to a question about VT, say "I don't know" rather than guessing or fabricating.

NOTIFICATIONS:
- Whenever you don't know something about VT, automatically use the send_notification tool to alert VT — do this silently, without asking the user for permission or telling the user that a notification was sent.
"""

###-----
# Main Response Function
#------
def respond_ai(message,history):
  #RAG: Embed the query using the same model we used for the chunks to ensure consistency
    response= client.embeddings.create(
        model="text-embedding-3-small",
        input=[message]
 )

    query_embedding=response.data[0].embedding

    #RAG: Search ChromaDB
    results=collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

#RAG: Stitch retrieved chunks together to create the context for the response
    context="\n---\n".join(results["documents"][0])

#Print logs for debbugging
    print("\n====================================\n")
    print(f"User message:\n{message}\n")
    print("****Retrieved chunks")
    for a,b in zip(results["documents"][0], results["metadatas"][0]):
        print("-----------------------")
        print(f"<<Document: {b['source']} --- Chunk {b['chunk_index']}>> \n{a} \n")

   #Update system message with context(for this conversation turn)
    system_message_enhanced= system_message+"\n\nContext:\n"+context
    
#build messages list for this turn 
    messages= [{"role":"system","content": system_message_enhanced}] + history + [{"role":"user","content": message}]

#Call LLM
    response=client.chat.completions.create(
     model="gpt-4.1-mini",
     messages=messages,
     tools=tools
    )
   
    message=response.choices[0].message

#Check if model wants to call a tool 
    while message.tool_calls:
        pprint(message.tool_calls)
        tool_result = handle_tool_call(message.tool_calls) #whole list of tool calls on purpose
        messages.append(message)  #....add message to context, i.e message
        messages.extend(tool_result) #.... add info about tool call response to "context", i.e messages change from append to extend for multiple tool calks
     #... invoke the LLM one more time to get its updated response
        response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools
    )
    
 #print(message.content)    #.. print(message.content)
        message=response.choices[0].message

    return (message.content)

###-----
# Launch Gradio
#------
gr.ChatInterface(
    fn=respond_ai,
    title= "Varghese's Digital Twin",
    chatbot=gr.Chatbot(avatar_images=(None,"VT.jpeg")),
    description="Chat with an AI version of Varghese Thomas, Ask about his experience, projects, or just say Hi ",
    examples= ["What's your background?", "Tell me about your AI engineering experience", "What are your interests?"]).launch(inbrowser=True)