# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns




# from riotwatcher import LolWatcher, ApiError
# import pandas as pd
# api_key = "RGAPI-bfe3ef1f-92d4-40b3-a1ae-c2ab246a9cbe"

# watcher = LolWatcher(api_key)
# my_region = 'na1'

# me = watcher.summoner.by_name(my_region, 'Doublelift')
# print(me)

# my_matches = watcher.match.matchlist_by_account(my_region, me['accountId'])
# last_match = my_matches['matches'][0]
# match_detail = watcher.match.by_id(my_region, last_match['gameId'])

# participants = []
# for row in match_detail['participants']:
#     participants_row = {}
#     participants_row['champion'] = row['championId']
#     participants_row['spell1'] = row['spell1Id']
#     participants_row['spell2'] = row['spell2Id']
#     participants_row['win'] = row['stats']['win']
#     participants_row['kills'] = row['stats']['kills']
#     participants_row['deaths'] = row['stats']['deaths']
#     participants_row['assists'] = row['stats']['assists']
#     participants_row['totalDamageDealt'] = row['stats']['totalDamageDealt']
#     participants_row['goldEarned'] = row['stats']['goldEarned']
#     participants_row['champLevel'] = row['stats']['champLevel']
#     participants_row['totalMinionsKilled'] = row['stats']['totalMinionsKilled']
#     participants_row['item0'] = row['stats']['item0']
#     participants_row['item1'] = row['stats']['item1']
#     participants.append(participants_row)
# df = pd.DataFrame(participants)
# df



# # check league's latest version
# latest = watcher.data_dragon.versions_for_region(my_region)['n']['champion']
# # Lets get some champions static information
# static_champ_list = watcher.data_dragon.champions(latest, False, 'en_US')

# # champ static list data to dict for looking up
# champ_dict = {}
# for key in static_champ_list['data']:
#     row = static_champ_list['data'][key]
#     champ_dict[row['key']] = row['id']
# for row in participants:
#     print(str(row['champion']) + ' ' + champ_dict[str(row['champion'])])
#     row['championName'] = champ_dict[str(row['champion'])]

# # print dataframe
# df = pd.DataFrame(participants)
# df



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from requests_html import HTMLSession
import time
session = HTMLSession()

def pull_summoner_names(url, start, end):
    website_string = session.get(url).html.html
    
    start_indices = [m.start() for m in re.finditer(start, website_string)]
    end_indices = [m.start() for m in re.finditer(end, website_string)]
    
    if (len(start_indices) != len(end_indices)):
        return('List lengths do not match.')
    
    result = [website_string[(start_indices[i]+len(start)):(end_indices[i])] for i in range(0,100)]
    return(result)

url = 'https://www.leagueofgraphs.com/rankings/summoners/{}'
regions = ['na', 'br', 'kr', 'euw', 'eune']
# regions = ['na']
start_tag = '<span class="name">'
end_tag = '</span>\n                                        <br/>\n'

top_summoners = dict()

for region in regions:
    top_summoners[region] = pull_summoner_names(url.format(region),start_tag,end_tag)

summoners = pd.DataFrame(top_summoners)
summoners

import numpy as np
from riotwatcher import LolWatcher 
api_key = "RGAPI-fd5f0c28-685d-4498-a2e6-43090cdffa7b"
lol_watcher = LolWatcher(api_key)
print('Success')
summoners.columns = ['na1', 'br1', 'kr', 'euw1', 'eun1']
summoners

def account_id_for_col(region, column):
    temp_list = []
    
    for summoner in column:
        try:
            temp_list.append(lol_watcher.summoner.by_name(region, summoner)['accountId'])
        except:
            print('Error for {}'.format(summoner))
            temp_list.append(np.nan)
        
    return(temp_list)
print('Success.')
print('NA1')
summoners['na1_account_id'] = account_id_for_col('na1', summoners.na1)
print('\nBR1')
summoners['br1_account_id'] = account_id_for_col('br1', summoners.br1)
print('\nKR')
summoners['kr_account_id'] = account_id_for_col('kr', summoners.kr)
print('\nEUW1')
summoners['euw1_account_id'] = account_id_for_col('euw1', summoners.euw1)
print('\nEUN1')
summoners['eun1_account_id'] = account_id_for_col('eun1', summoners.eun1)


match_columns = ['teamId', 'win', 'firstBlood', 'firstTower', 'firstInhibitor',
       'firstBaron', 'firstDragon', 'firstRiftHerald', 'towerKills',
       'inhibitorKills', 'baronKills', 'dragonKills', 'vilemawKills',
       'riftHeraldKills', 'dominionVictoryScore', 'bans', 'region']
match_df = pd.DataFrame(columns = match_columns)
match_df
account_id_columns = ['na1_account_id', 'br1_account_id', 'kr_account_id', 'euw1_account_id', 'eun1_account_id']
# account_id_columns = ['na1_account_id']


#under test situation range turns to 10 

for column in account_id_columns:
    
    current_column = summoners[column]
    current_region = column.split('_')[0]
    print('Starting {}....'.format(current_region))
    
    for index in range(0,100):
        
        if type(summoners[column].iloc[index]) == float:
            print('Skipping index {}, column {} because nan value.'.format(index, column))
            continue
            
        try:    
            temp_game_ids = [game['gameId'] for game in lol_watcher.match.matchlist_by_account(current_region, summoners[column].iloc[index])['matches']]
        except:
            print('Skipping games at index {}, column {}.'.format(index, column))
            continue
        
        for gameid in temp_game_ids[:10]:
            try:
                temp_df = pd.DataFrame(lol_watcher.match.by_id(current_region, gameid)['teams'])
                temp_df['time'] = lol_watcher.match.by_id(current_region,gameid)['gameDuration']
                temp_df['region'] = [current_region] * len(temp_df)
                temp_df['gameId'] = [gameid]*len(temp_df)
                match_df = pd.concat([match_df, temp_df], sort = False)
            except:
                print('Skipping game at index {}, column {}'.format(index, column))
                continue

    print('Finished {}'.format(current_region))
    print('Current length of dataframe: {}\n'.format(len(match_df)))
match_df['gold']=[0]*len(match_df)
gameId_list = []

# for gameId in match_df['gameId']:
#     gameId_list.append(int(gameId))
# result_gold = [0] * len(match_df)

# match_df = match_df.reset_index()
# match_df['gold'] = [0]* len(match_df)

for index in range(0,len(match_df),2):
    gameinfo = lol_watcher.match.by_id(match_df['region'][index],int(match_df['gameId'][index]))
    for team_100 in range(5):
        match_df['gold'][index] += gameinfo['participants'][team_100]['stats']['goldEarned']
    print('finish',match_df['region'][index],index)
    time.sleep(0.75)
    for team_200 in range(5,10):
        match_df['gold'][index+1] += gameinfo['participants'][team_200]['stats']['goldEarned']
    print('finish',match_df['region'][index+1],index+1)
    time.sleep(0.75)






for index in range(9192,len(match_df),2):
    gameinfo = lol_watcher.match.by_id(match_df['region'][index],int(match_df['gameId'][index]))
    for team_100 in range(5):
        match_df['gold'][index] += gameinfo['participants'][team_100]['stats']['goldEarned']
    print('finish',match_df['region'][index],index)
    time.sleep(0.75)
    for team_200 in range(5,10):
        match_df['gold'][index+1] += gameinfo['participants'][team_200]['stats']['goldEarned']
    print('finish',match_df['region'][index+1],index+1)
    time.sleep(0.65)




# for current_region in summoners.columns[:5]:
#     for i in range(0,len(match_df),2):
#         gameinfo = lol_watcher.match.by_id(current_region,gameId_list[i])

#         for team_100 in range(5):
#             result_gold[i] += gameinfo['participants'][team_100]['stats']['goldEarned']
#         print("finish",i)
#         time.sleep(1)
#     # time.sleep(5)
#         for team_200 in range(5,10):
#             result_gold[i+1] += gameinfo['participants'][team_200]['stats']['goldEarned']
#         print("finish",i+1)
#         time.sleep(1)
#     # time.sleep(5)    
# match_df['gold'] = result_gold







    # for team_100 in range(5):
    #     result_gold[i] += lol_watcher.match.by_id(current_region,gameId_list[i])['participants'][team_100]['stats']['goldEarned']
    # for team_200 in range(5,10):
    #     result_gold[i+1] += lol_watcher.match.by_id(current_region,gameId_list[i+1])['participants'][team_200]['stats']['goldEarned']

# match_df['gold'] = result_gold

# match_df['time'] = [0]*len(match_df)

# test part
# test_gamelist  = lol_watcher.match.matchlist_by_account("na1",summoners['na1_account_id'][0])['matches']  # 这边return 出来的是 一个 对局的资料 
# test_game_id = [game['gameId'] for game in test_gamelist]
# test_game_example = test_game_id[0]

# example_totalinfo = lol_watcher.match.by_id("na1",test_game_id[0])
# example_totaltime = example_totalinfo['gameDuration']
# test_df = pd.DataFrame(lol_watcher.match.by_id(current_region,test_game_example)['teams'])
# temp_100_gold = 0
# temp_200_gold = 0 
# for i in range(5):
#     temp_100_gold += example_totalinfo['participants'][i]['stats']['goldEarned']
# for j in range(5,10):
#     temp_200_gold += example_totalinfo['participants'][i]['stats']['goldEarned']

# test_gold =dict(gold = temp_100_gold,gold2 = temp_200_gold)
# double_game_detail = lol_watcher.match.by_id('na1',match_id)
# double_game_detail['participants'][0]
# double_game_detail['participants']['championId'==236]
