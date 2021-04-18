#this script extracts the top level comments of all the videos(if comments are abled for them) in a youtube channel provided by you (the channel Id)
#you must also provide the access token
#all the interactions you will have with this script will be through the mainfunction() call in the line 231, except the access token in line 15



from googleapiclient.discovery import build
import pandas as pd
from csv import writer
import sys

sys.stdout.reconfigure(encoding='utf-8')

apikey = ''
#put the access token here (string)

youtube = build('youtube', 'v3', developerKey = apikey)

print('Build: Done\n')



#-------------------------------------------------


def checkvideo(videoId, videoIdFile = ''):
    #checking if the csv file is umpty (doesnt have header)
    line3 = None
    with open(videoIdFile, 'r', encoding='utf-8') as file0:
        line = file0.readline()
        line1 = line + file0.readline()
        line2 = line1 + file0.readline()
        line3 = line2 +file0.readline()
    if len(line3) == 0:
        with open (videoIdFile,'a+', encoding='utf-8') as file:
            csv_writer=writer(file)
            csv_writer.writerow(['VideoID'])

    videodf=pd.read_csv(videoIdFile)
    condition = (videodf['VideoID']==videoId).sum()
    if condition ==0:
        return 0
    else: 
        return 1



#------------------------------------
def getComments(videoTitle, videoDescription, channelId, C, part=['snippet'], maxResults=100, textFormat='plainText', order='time', videoId='', csvFile ='', videoIdFile=''):
    

    #checking if the video has already been used
    used=checkvideo(videoId, videoIdFile=videoIdFile)
    if used == 0:
        print('video: ',videoId,' not used before\nproceeding to extraction...\n')
    else:
        # in the case of C == S (skip already gotten vids)
        if C in['S','s']: 
            print('\'S\': video already used, skipping video')
            return None
        # in the case of C == A (get already gotten video again)
        if C in['A','a']:
            print('\'A\': video already used, getting comments anyways')



    #adding videoId to videoIdFile:
    print('adding videoId to videoIdFile...')
    with open (videoIdFile,'a+', encoding='utf-8') as file:
        csv_writer=writer(file)
        csv_writer.writerow([videoId])






    print('proceeding to extraction of comments and other data...')
    #preparing the lists
    comments,commentsId,videoIDS,parentIDS, videoTitles, videoDescriptions, channelIds=[],[],[],[],[],[],[]
    #making an api call
    try:
        response= youtube.commentThreads().list(part=part, maxResults=maxResults, textFormat=textFormat, order=order, videoId=videoId).execute()
    except:
        print('an error occured, maybe the comments are disabled on this video, moving to next video if any...')
        return None


    totalResults = response['pageInfo']['totalResults']

    print('response from youtube API gotten, proceeding to getting the relevant data...')
    #a loop that will continue until we run out of quoata:
    while response:
        for item in response['items']:
            comment=item['snippet']['topLevelComment']['snippet']['textDisplay']
            commentId=item['snippet']['topLevelComment']['id']
            videoID=item['snippet']['videoId']
            parentID='None'
            


            #append to list
            comments.append(comment)
            commentsId.append(commentId)
            videoIDS.append(videoID)
            parentIDS.append(parentID)
            videoTitles.append(videoTitle)
            videoDescriptions.append(videoDescription)
            channelIds.append(channelId)
            
            #checking if the csv file is umpty
            with open(csvFile, 'r', encoding='utf-8') as file0:
                line = file0.readline()
                line1 = line + file0.readline()
                line2 = line1 + file0.readline()
                line3 = line2 +file0.readline()
                if len(line3) == 0:
                    with open (csvFile,'a+', encoding='utf-8') as file:
                        csv_writer=writer(file)
                        csv_writer.writerow(['comment', 'commentId', 'parentCommentID', 'videoTitle', 'videoID', 'videoDescription', 'channelId'])

            #write to csv line by line
            with open (csvFile,'a+', encoding='utf-8') as file:
                csv_writer=writer(file)
                csv_writer.writerow([comment, commentId, parentID, videoTitle, videoID, videoDescription, channelId])


        #check for nextPageToken and if it exists, set response equal to it
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(part=part, maxResults=maxResults, textFormat=textFormat, order=order, videoId=videoId, pageToken = response['nextPageToken']).execute()
            totalResults += response['pageInfo']['totalResults']
            print('new response with nextPageToken gotten, proceeding to getting the relevant data...')
        else:
            break
    else: print('--- max quota reached ---\n')

    print('\ntotal Results gotten from this video:', totalResults, '\n\n')   
    
    return totalResults
    #return totalResults, {'comments': comments, 'commentsId': commentsId, 'parentCommentIDS':parentIDS, 'videoTitles': videoTitles, 'videoIDS':videoIDS, 'videoDescriptions': videoDescriptions, 'channelIds':channelIds}




#-----------------------

def mainfunction(channelid= '', channelCsv='', videoIdFile = '', csvFile=''):

    #checking if the channels csv file is umpty (doesnt have header)
    line3 = None
    with open(channelCsv, 'r', encoding='utf-8') as file0:
        line = file0.readline()
        line1 = line + file0.readline()
        line2 = line1 + file0.readline()
        line3 = line2 +file0.readline()
    if len(line3) == 0:
        #adding a header to the csv file
        with open (channelCsv,'a+', encoding='utf-8') as file:
            csv_writer=writer(file)
            csv_writer.writerow(['ChannelId'])
    #checking for matching channelIds with the new one
    channelCsvdf=pd.read_csv(channelCsv)
    condition = (channelCsvdf['ChannelId']==channelid).sum()
    R = None
    if condition ==0:
        print('Channel: ',channelid,' not used before. do you wanna proceed ?\n')
        R = input('y\\n?')
        if R in['n','N']: return None #if this doesnt work, try sys.exit(0)
        if R in['y','Y']: print('proceeding...\n')
        #adding videoId to videoIdFile:
        print('adding Channelid to ChannelIdFile...')
        with open (channelCsv,'a+', encoding='utf-8') as file:
            csv_writer=writer(file)
            csv_writer.writerow([channelid])

    else: 
        print('Channel already used.')
        R = input('Enter \'E\' if you wanna exit, \'S\' if you wanna skip the already gotten videos and only get the rest (if exists), \'A\' if you wanna extract everything again: ')
        if R in ['e','E']: return None
        elif R in['s','S']: print('proceeding to getting rest of video Ids...')
        elif R in['A','a']: 
            S = input('are you sure (y/n) ? their might be duplicates: ')
            if S in['N','n']: return None
            if S in['Y','y']: print('proceeding to getting All video Ids...')
        else: 
            print('you entered a wrong character')
            return None



    #if R == A or S:
    channel = youtube.channels().list(part = 'contentDetails', id = channelid).execute()   
    
    item = channel['items'][0]
    uploads = item['contentDetails']['relatedPlaylists']['uploads'] 
    print('    - - - uploads playlist gotten')

    playlistItem = youtube.playlistItems().list(part='snippet', playlistId=uploads, maxResults=50).execute() 
    print('    - - - videoIds and other data gotten')


    totalResults = 0 #to calculate the total nbr of comments gotten from this channel (without counting the skiped ones if exists)
    totalVids = 0 #to calculate the total nbr of videos getten from the channel
    while playlistItem:
        for item in playlistItem['items']:
            videoTitle = item['snippet']['title']
            videoDescription = item['snippet']['description']
            channelId = item['snippet']['channelId']
            videoId = item['snippet']['resourceId']['videoId'] 
            tempResults = getComments(videoId=videoId, videoTitle=videoTitle, videoDescription=videoDescription, channelId=channelId, videoIdFile= videoIdFile, csvFile=csvFile, C=R) 
            #if you want to return the dictionaries of the comments gotten this time uncomment other return call in getcomments()
            if tempResults not in [0,None]:
                totalResults +=tempResults
                totalVids +=1

        if 'nextPageToken' in playlistItem:
            playlistItem = youtube.playlistItems().list(part='snippet', playlistId=uploads, maxResults=5, pageToken = playlistItem['nextPageToken']).execute()
            print('    - - - New response with nextPageToken gotten, proceeding to getting the other videoIds et al...')
        else:
            break   

    else: print('--- max quota reached ---\n')

    print('\n\n    - - - Total nbr of videos getten: ', totalVids)
    print('\n\n    - - - Total nbr of comments getten this time from the channel: ', totalResults)



#------------------------------
mainfunction(channelid= '', 
        channelCsv='',
        videoIdFile = '',
        csvFile='')

#dataframe = pd.read_csv()
#print(dataframe)
