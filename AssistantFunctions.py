from langchain_community.tools import DuckDuckGoSearchResults
import os
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool
from PIL import Image
import base64
from io import BytesIO
import requests
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from msal import ConfidentialClientApplication
import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.keyvault.secrets import SecretClient
from langchain_core.tools import tool
import base64

#########################################
# TOOLS #
#########################################
@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    return: a * b
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    return: a + b
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    return: a / b
    """
    return a / b


@tool
def web_search(question: str) -> str:
    """Searches the web for the question.

    Args:
        question: question to search for
    return: search results
    """
    search = DuckDuckGoSearchResults(output_format="list")
    return search.invoke(question)


def get_access_token():
    """
    Get an access token for Microsoft Graph APIs using Azure Key Vault secrets.
    :param user_id: The user ID.
    :return: The access token if successful, otherwise an error message.
    """
    load_dotenv()

    #VAULT_URL = os.environ["VAULT_URL"]
    #credential = DefaultAzureCredential()
    #key_vault_client = SecretClient(vault_url=VAULT_URL, credential=credential)
    #client_id = key_vault_client.get_secret("meeting-assistant-client-id", version="5ee4f687b030409d9c8c74be1806185f").value
    #client_secret = key_vault_client.get_secret("meeting-assistant-client-secret", version="da36a03ce389423a9728d8d20414efb3").value
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    tenant_id = os.getenv("TENANT_ID")
    #tenant_id = key_vault_client.get_secret("meeting-assistant-tenant-id").value

    auth = f"https://login.microsoftonline.com/{tenant_id}"
    scope = "https://graph.microsoft.com/.default"
    
    app = ConfidentialClientApplication(
        client_id=client_id,
        client_credential=client_secret,
        authority=auth
    )
    
    result = app.acquire_token_for_client(scopes=[scope])

    if "access_token" in result:
        access_token = result["access_token"]
        print("Access Token:", access_token)
        return access_token 
    else:
        print(f"Failed to acquire token, {result}")
        return f"Failed to acquire token,{result}"
    

def get_user_details(user_id: str):
    """
    Get user details from Microsoft Graph APIs.
    :param user_id: The user ID.
    :return: The user details if successful, otherwise an error message.
    """
    access_token = get_access_token()

    # Create an HTTP client and call the Fabric APIs
    base_url = f"https://graph.microsoft.com/v1.0/users/{user_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    response = requests.get(base_url, headers=headers)
    print("User Details Log: ", response.json())
    response_body = response.text
    return response_body


@tool
def get_all_meetings(user_id: str, date=None):
    """
    Get all meetings for a user from Microsoft Graph APIs.
    :param user_id: The user ID.
    :param date: The date to filter the meetings.
    :return: The meetings if successful, otherwise an error message.
    """
    access_token = get_access_token()

    if date:
        start_time = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
        end_time = (datetime.strptime(start_time, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_time = time.strftime("%Y-%m-%d", time.gmtime(time.time()))
        end_time = time.strftime("%Y-%m-%d", time.gmtime(time.time() + 1 * 24 * 60 * 60))

    base_url = f"https://graph.microsoft.com/v1.0/users/{user_id}/calendarView?startDateTime={start_time}&endDateTime={end_time}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Prefer": "outlook.timezone=\"Turkey Standard Time\""
    }

    # Call list events API with query parameters
    response = requests.get(base_url, headers=headers)

    if response.status_code != 200:
        return f"Failed to get all meetings, {response.text}", access_token
    response_body = response.json()
    print("Meeting List Log: ", response_body)

    meeting_list = [{"subject": meeting["subject"], "startDateTime": meeting["start"]["dateTime"], "endDateTime": meeting["end"]["dateTime"], "isOnlineMeeting": meeting["isOnlineMeeting"], "onlineMeeting": meeting["onlineMeeting"], "attendees": meeting["attendees"], "organizer": meeting["organizer"]} for meeting in response_body["value"]]
    
    return meeting_list, access_token


def get_online_meeting_IDs(user_id: str, date=None):
    """
    Get all online meeting IDs for a user from Microsoft Graph APIs.
    :param user_id: The user ID.
    :return: The online meeting IDs if successful, otherwise an error message.
    """
    meeting_id_list = []

    if date is not None:
        meeting_list, access_token = get_all_meetings(user_id, date)
    else:
        meeting_list, access_token = get_all_meetings(user_id)
    
    if type(meeting_list) == str:
        return meeting_list, access_token
    
    for meeting in meeting_list:
        if meeting["isOnlineMeeting"]:
            join_url = meeting["onlineMeeting"]["joinUrl"]
            meeting_id_endpoint = f"https://graph.microsoft.com/v1.0/users/{user_id}/onlineMeetings?$filter=joinWebUrl%20eq%20'{join_url}'"
            meeting_id_request = requests.get(meeting_id_endpoint, headers={"Authorization": f"Bearer {access_token}"}) 
            print("Meeting ID Log: ", meeting_id_request.json())
            if meeting_id_request.status_code == 200:
                meeting_id_list.append({"meetingID": meeting_id_request.json()["value"][0]["id"], "subject": meeting_id_request.json()["value"][0]["subject"], "joinUrl": join_url, "startDateTime": meeting_id_request.json()["value"][0]["startDateTime"], "endDateTime": meeting_id_request.json()["value"][0]["endDateTime"], "participants": meeting_id_request.json()["value"][0]["participants"]})

    return meeting_id_list, access_token


def get_meeting_transcript_urls(user_id: str, date=None):
    """
    Get all meeting transcript content URLs for a user from Microsoft Graph APIs.
    :param user_id: The user ID.
    :return: The meeting transcript content URLs if successful, otherwise an error message.
    """
    if date is not None:
        meeting_id_list, access_token = get_online_meeting_IDs(user_id, date)
    else:
        meeting_id_list, access_token = get_online_meeting_IDs(user_id)

    if type(meeting_id_list) == str:
        return meeting_id_list, access_token
    
    transcript_content_url_list = []

    for meeting_id in meeting_id_list:
        transcript_endpoint = f"https://graph.microsoft.com/v1.0/users/{user_id}/onlineMeetings/{meeting_id['meetingID']}/transcripts"
        transcript_request = requests.get(transcript_endpoint, headers={"Authorization": f"Bearer {access_token}"})
        print("Transcript URL Log: ", transcript_request.json())
        if transcript_request.status_code == 200 and transcript_request.json()["@odata.count"] > 0:
            for transcript_url in transcript_request.json()["value"]:
                transcript_content_url_list.append({"meetingID": meeting_id['meetingID'], "joinUrl": meeting_id['joinUrl'], "subject": meeting_id['subject'], "transcriptContentUrl": transcript_url["transcriptContentUrl"], "startDateTime": meeting_id['startDateTime'], "endDateTime": meeting_id['endDateTime'], "createdDateTime": transcript_url["createdDateTime"], "participants": meeting_id['participants']})
    
    return transcript_content_url_list, access_token


@tool
def get_meeting_transcript_contents(user_id: str, subject: str, date=None):
    """
    Get all meeting transcript contents for a user from Microsoft Graph APIs.
    :param user_id: The user ID.
    :param subject: The subject of the meeting.
    :param date: The date to filter the meetings.
    :return: The meeting transcript contents if successful, otherwise an error message.
    """

    if date is not None:
        transcript_content_url_list, access_token = get_meeting_transcript_urls(user_id, date)
    else:
        transcript_content_url_list, access_token = get_meeting_transcript_urls(user_id)
    
    if type(transcript_content_url_list) == str:
        return transcript_content_url_list, access_token
    
    transcript_contents = []

    for transcript_content_url in transcript_content_url_list:
        print(f"{subject.lower()} in {transcript_content_url['subject'].lower()}: {subject.lower() in transcript_content_url['subject'].lower()} && {date} in {transcript_content_url['createdDateTime']}: {date in transcript_content_url['createdDateTime']}")
        if subject.lower() in transcript_content_url["subject"].lower() and date in transcript_content_url["createdDateTime"]:
            transcript_content_request = requests.get(f"{transcript_content_url['transcriptContentUrl']}?$format=text/vtt", headers={"Authorization": f"Bearer {access_token}"})
            transcript_content_request.encoding = "utf-8"
            print("Transcript Content Log: ", transcript_content_request.status_code)
            print(f"{subject.lower()} in {transcript_content_url['subject'].lower()}: {subject.lower() in transcript_content_url['subject'].lower()}")

            if transcript_content_request.status_code == 200:
                transcript_contents.append(transcript_content_request.text)
            elif transcript_content_request.status_code != 200:
                print(transcript_content_request.json())
                transcript_contents.append(f"Failed to get {subject} transcript content, {transcript_content_request.json()}")
    
    if transcript_contents == []:
        return f"{subject} toplantısıyla ilgili transkript bulunamadı."
    return " ".join(transcript_contents)
