import sys
import pandas as pd
import langdetect

sys.stdout.reconfigure(encoding='utf-8')


                            #ps: most of the following code has been made by the ideea in mind that we will only work with 450000 comments max in our dataframe, so for future work, and in case you dont have enaugh ram, make sure to use chunksize (and while modifying the code, be sure that while saving the chunks in the csv the indexing doesnt go back to zero after each chunk, i think that problem occurs when reading from multiple files)

#start by giving the necessary csv files:

#the csv that will contain all our comments
rawDataFile = 'cleanData4/rawAllInOneSummarized.csv'
#the csv that will contain the result after removal of latin characters (french and arabizi comments for exemple)
noLatinFile = 'cleanData4/noLatin.csv'
#after removal of MSA (standard modern arabic)
noMsaFile = 'cleanData4/noMSA.csv'
#removal of unwanted special characters
noSpecialCharsFile = 'cleanData4/afterSpecialCharRemoval.csv'
#after putting spaces between nbrs and words, emojis and words, emojis and emojis. so that they will be considered separte tokens by our model
spacingFile = 'cleanData4/afterSpacing.csv'
#removal of redundant punctuation (except some)
rmvdRedundantPunctFile = 'cleanData4/afterRemovalOfRedundantPunct.csv'
#normalization of the redundant punctuations that we didnt want removed as well as stressed words
normalizationFile = 'cleanData4/afterNormalization.csv'
#turning imojis in a representation with characters that arabert can read
emojiFormatingFile = 'cleanData4/afterEmojiFormating.csv'
#some methods of cleaning that Farassa (or arabert, dont remember) can handle
farassaMethodsFile = 'cleanData4/afterFarassaCleaning.csv'
#droping duplicates in comments as well as comments that became umpty after all the cleaning
finalTouchesFile = 'cleanData4/afterFinalTouches.csv'



#now give a list of all the csvs that you have that contain comments
commentsCsvs = ['comments14.csv']#,'comments2.csv','comments3.csv','comments4.csv',
#'comments5.csv','comments6.csv','comments7.csv','comments8.csv','comments9.csv',
#'comments10.csv','comments11.csv','comments12.csv','comments13.csv','comments14.csv','comments15.csv',
#'comments16.csv','comments17.csv','csvfile.csv','csvfile2.csv','csvfile3.csv','csvfile4.csv',
#'csvfile5.csv','cscfile6.csv','csvfile7.csv']


'''


'''
#-------------------------------------------------------------------------------------


##putting all the csv's in one summarised(keeping just the comment, videoId, and channelId, since its all am gonna use this time)
##ps: in this part of the cleaning, we didnt put index_col = 0 in pandas. thats because when we extracted the comments and put them in the csvs, we didnt use pandas, so their is no index column
##ps2: since we dont have time, am only gonna take 450000 comments, to result in approximatly 200000 comments clean                    

y = 1
chunksize = 18750
for csv in commentsCsvs:
                #first file (for the header)
    if y == 1:
        for chunk in pd.read_csv(csv, chunksize=chunksize):
            print(chunk)
            newChunk = chunk[['comment','videoID','channelId']]
            print("headerrrrrrrrrrrrrrrrrrrrrrrrrr")
            newChunk.to_csv(rawDataFile, mode='a', header=True)
            break
        y = 2
        continue
                    #second file and onwards
    if y == 2:    
        for chunk in pd.read_csv(csv, chunksize=chunksize):
            print(chunk)
            newChunk = chunk[['comment','videoID','channelId']]
            print("noooooooooooooooooooooooo header")
            newChunk.to_csv(rawDataFile, mode='a', header=False)
            break
mydf = pd.read_csv(rawDataFile, index_col=0)
mydf.reset_index(drop=True, inplace=True)
mydf.to_csv(rawDataFile, mode='w', header=True)
print(mydf)

#----------------------------------------------------------------------------------------------

                    #removing Latin chars 

AtoZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
def dropLatin(row):
    try:
        for letter in row["comment"]:
            if letter in AtoZ:
                chunk.drop(row.name,inplace = True)
                break
            else:
                continue
    except:
        chunk.drop(row.name,inplace = True)


#------------------------------------------------------------------------------------------------------------------------------------------

                                        #Removing MSA comments

#this func is attempts to separate the arabic comments (MSA) for the dz dialect ones
#ps: the comments that contain only imojis or nbrs (in latin) will be considered dz lang, only bacause its the dz lang database that i will work with so i put them there since i need them.




def MSAandALGDseparator(row):
    #to detect if we found at least one word in the comment that doesnt contain only imojis or nbrs, cause those that contain only imojis or nbrs return an error in lang detect, so after the try/except i wanna make sure to know that all the words in that comment have only imojis or nbrs so that i write code that puts it in dz category
    imoji = 1
    prob = None
    #we split the comment into words, so that we run the langdetect on each word separatly 
    SplitCom = row["comment"].split(" ")
    for word in SplitCom:
        #try/except in case the word has only imojis or nbrs
        try:
            mylangdetect = list(langdetect.detect_langs(word))
        except:
            print('only imojis, skipping this token')
            continue
        #in case we did find at least one word with not only imojis or nbrs in it, we flag it with zero
        imoji = 0
        #si langdetect detect plus qu'une langue, je suppose que la probabilites de l'arabe n'est pas presque certaine, donc je suppose que le commentaire est en langue algerienne
        if len(mylangdetect) > 1:
            row["lang"] = "dz"
            print('dz', '   ,more than one lang detected')
            return 
        else:
            lang = str(mylangdetect[0])
            #the first two chars are the language symbol (ar, fa, en, fr..etc)
            langName = lang[0:2]
            #if the language isnt arabic, it means its dz lang (ps: thats because i have already cleaned all the comments written in latin)
            if langName != 'ar':
                row["lang"] = "dz"
                print('dz', "   lang detected not arabic")
                return
            #if it detected arabic, am gonna see if the probability is low enaugh. i noticed that for arabic comments (and infortunatly even for many dz lang ones, it detects the probability of arabic as 0.99999...(five nines and then smth) so if its lower than that or equal to 0.999990 am gonna suppose its in dz lang)
            else:
                #the rest (after the first two) of the chars are the probability
                prob = float(lang[3:])
                if prob <= 0.999990:
                    row["lang"] = "dz"
                    print('dz','  prob under threshold, = ', prob)
                    return
                else: continue
    #the else statement of the for loop
    else:
        #if all the content of the comments is imojis or nbrs, we keep it in our algerian lang database 
        if imoji ==1: 
            row["lang"] = "dz"
            print('dz,   comment only has imojis or nbrs')
        else:
            row["lang"] = "ar"
            print('ar','probability = ', prob)



#----------------------------------------------------------------------------------------------


    ##execution of all except the first (putting all in one csv)

# Latin char removal:
x = 0
chunksize = 50000
print("\n\n\nstarting removal of Latin comments\n\n")
for chunk in pd.read_csv(rawDataFile, chunksize=chunksize, index_col=0):
    chunk.apply(dropLatin, axis=1)
    chunk.reset_index(drop=True, inplace=True)
    print(chunk)
    if x==0:
        print("headerrrrrrrrrrrrrrrrrrrrrrrrrr")
        chunk.to_csv(noLatinFile, mode='a', header=True)
        x=1
    else:
        print("noooooooooooooooooooooooo header")
        chunk.to_csv(noLatinFile, mode='a', header=False)
mydf = pd.read_csv(noLatinFile, index_col=0)
mydf.reset_index(drop=True, inplace=True)
mydf.to_csv(noLatinFile, mode='w', header=True)







    ##MSA removal:

print("\n\n\nstarting removal of MSA comments\n\n")
mydf = pd.read_csv(noLatinFile, index_col=0)
#creating a column for the language label
mydf["lang"]=""
#applying the function
mydf.apply(MSAandALGDseparator, axis=1)
#putting dz ones in a dataframe
dzDF = mydf[mydf["lang"] == 'dz']
dzDF.reset_index(drop=True, inplace=True)
#putting them in separate csvs, this is in case the csvs are umpty (first use)
dzDF.to_csv(noMsaFile, mode='w', header=True)

