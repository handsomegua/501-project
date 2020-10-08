import requests
import json
import numpy
import pandas as pd
from sklearn.manifold import MDS
from sklearn import preprocessing
import csv
import seaborn as sns
import nltk
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.cluster import AgglomerativeClustering
import sklearn
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.metrics import euclidean_distances, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 
response = requests.get('http://ddragon.leagueoflegends.com/cdn/10.20.1/data/en_US/champion.json')
response.json()
allchamps = []
champRawData = json.loads(response.text)
crd = champRawData['data']
for i in crd:
    name = crd[i]['id']
    champID = crd[i]['key']
    ADbase = crd[i]['stats']['attackdamage']
    ADpl = crd[i]['stats']['attackdamageperlevel']
    FMS = crd[i]['stats']['movespeed']
    ASPL= crd[i]['stats']['attackspeedperlevel']
    Title = crd[i]['title']
    blurb = crd[i]['blurb']
    tags = crd[i]['tags'][0]
    hp = crd[i]['stats']['hp']
    row = [name,champID,ADbase,ADpl,FMS,ASPL,Title,blurb,tags,hp]
    allchamps.append(row)
DF = pd.DataFrame(allchamps)
DF.columns = ['name','champID','ADbase','ADpl','FMS','ASPL','Title','blurb','tags','hp']
DF.to_csv("DF_501.csv")
# for i in range(151):
#     textfile = open("/Users/hayashishishin/new_text_data/text_data%d.txt"%(i),'w')
#     textfile.write(DF['blurb'][i])
#     textfile.close()
# load record data 

#check the number of tags in tag_list
tag_list = []
for i in DF['tags']:
    if i in tag_list:
        continue
    else:
        tag_list.append(i)

record_df = DF.drop(columns = ['Title','tags','champID','name','blurb'])
# record_df = preprocessing.scale(record_df)
vectorize_record_df = preprocessing.scale(record_df) # we could choose to use vectorize or not
#silhouette
sil = []
for k in range(2,7):
    kmeans = KMeans(n_clusters = k).fit(record_df)
    preds = kmeans.fit_predict(record_df)
    sil.append(silhouette_score(record_df,preds,metric = 'cosine'))
plt.plot(range(2, 7), sil)
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sil')
# plt.savefig('sil.png')
plt.show()

#clustering numerical data using k means  
kmeans_objectrecord = sklearn.cluster.KMeans(n_clusters = 3)
kmeans_record = kmeans_objectrecord.fit(record_df)
cluster_result_record_data = kmeans_record.labels_
record_df['cluster'] = cluster_result_record_data
DF['cluster'] = cluster_result_record_data
#calculate different distance 
cosdist = 1 - cosine_similarity(record_df)
eudist = euclidean_distances(record_df)
mandist = manhattan_distances(record_df)
#visualization for k means part
#2d
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
pos_cos = mds.fit_transform(cosdist)
pos_eu = mds.fit_transform(eudist)
pos_man = mds.fit_transform(mandist)
#Plot under cos distance
xs_cos, ys_cos = pos_cos[:,0],pos_cos[:,1]
for x,y,name in zip(xs_cos,ys_cos,DF['name']):
    plt.scatter(x,y)
    plt.text(x,y,name)
# plt.savefig('2d_cos.png')
plt.show()
#Plot under euclidean distance 
xs_eu,ys_eu = pos_eu[:,0],pos_eu[:,1]
for x,y,name in zip(xs_eu,ys_eu,DF['name']):
    plt.scatter(x,y)
    plt.text(x,y,name)
plt.savefig('2d_eu.png')
plt.show()
#plot under manhattan distance
xs_man,ys_man = pos_man[:,0],pos_man[:,1]
for x,y,name in zip(xs_man,ys_man,DF['name']):
    plt.scatter(x,y)
    plt.text(x,y,name)
# plt.savefig('2d_man.png')
plt.show()


#3d
mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
pos_cos = mds.fit_transform(cosdist)
pos_eu = mds.fit_transform(eudist)
pos_man = mds.fit_transform(mandist)
# 3d base on cos dist
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for x, y, z, s in zip(pos_cos[:, 0], pos_cos[:, 1], pos_cos[:, 2], DF['name']):
    plt.scatter(x,y,z)
    ax.text(x, y, z, s)

# ax.set_xlim3d(-.2,.3) #stretch out the x axis
# ax.set_ylim3d(-.2,.3) #stretch out the x axis
# ax.set_zlim3d(-.2,.3) #stretch out the z axis
# plt.savefig('3d_cos.png')
plt.show()
# # 3d base on euclidean distance
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos_eu[:, 0], pos_eu[:, 1], pos_eu[:, 2])
for x, y, z, s in zip(pos_eu[:, 0], pos_eu[:, 1], pos_eu[:, 2], DF['name']):
    plt.scatter(x,y,z)
    ax.text(x, y, z, s)

# ax.set_xlim3d(-.2,.3) #stretch out the x axis
# ax.set_ylim3d(-.2,.3) #stretch out the x axis
# ax.set_zlim3d(-.2,.3) #stretch out the z axis
# plt.savefig('3d_eu.png')
plt.show()
# #3d base on manhattan distance
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos_man[:, 0], pos_man[:, 1], pos_man[:, 2])
for x, y, z, s in zip(pos_man[:, 0], pos_man[:, 1], pos_man[:, 2], DF['name']):
    plt.scatter(x,y,z)
    ax.text(x, y, z, s)

# ax.set_xlim3d(-.2,.3) #stretch out the x axis
# ax.set_ylim3d(-.2,.3) #stretch out the x axis
# ax.set_zlim3d(-.2,.3) #stretch out the z axis
# plt.savefig('3d_man.png')
plt.show()

#ward clustering using cosdist
linkage_matrix_cos = ward(cosdist)   #i don't know why here the len(linkage_matrix) is 150 not 151
fig, ax = plt.subplots(figsize=(15, 20))
dendrogram(linkage_matrix_cos)
# plt.savefig('dendrogram_cosdist.png')
plt.show()
#ward clustering using euclidean distance
linkage_matrix_eu = ward(eudist)   #i don't know why here the len(linkage_matrix) is 150 not 151
fig, ax = plt.subplots(figsize=(15, 20))
dendrogram(linkage_matrix_eu)
# plt.savefig('dendrogram_eudist.png')
plt.show()
#ward clustering using manhattan distance
linkage_matrix_man = ward(mandist)   #i don't know why here the len(linkage_matrix) is 150 not 151
fig, ax = plt.subplots(figsize=(15, 20))
dendrogram(linkage_matrix_man)
# plt.savefig('dendrogram_mandist.png')
plt.show()

#heatmap
cmap = sns.diverging_palette(220, 20, l = 40, s = 99, sep = 20, center = 'light', as_cmap = True) 
sns.heatmap(record_df.corr(), vmin = 0, vmax = 1, annot = True, cmap = cmap, lw = .5, linecolor = 'white')



# load text data
text_data = DF['blurb']
stopwords = set(STOPWORDS) 
wordcloud_list = []
for i in range(text_data.shape[0]):
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(text_data[i]) 
    wordcloud_list.append(wordcloud)
plt.imshow(wordcloud)
# plt.savefig('wordcloud_sample.png')
myVector_count = CountVectorizer(input='content')
DTM_count = myVector_count.fit_transform(text_data)

columnNames = myVector_count.get_feature_names()
df_text = pd.DataFrame(DTM_count.toarray(),columns = columnNames)
#normaalizing via scaling min max
x = df_text.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_text_scaled = pd.DataFrame(x_scaled)
#sil determine k
sil = []
for k in range(2,6):
    kmeans = KMeans(n_clusters = k).fit(df_text_scaled.values)
    preds = kmeans.fit_predict(df_text_scaled.values)
    sil.append(silhouette_score(df_text_scaled.values,preds,metric = 'euclidean'))
plt.plot(range(2, 6), sil)
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sil')
# plt.savefig('sil_textdata.png')
plt.show()
#kmeans 
kmeans_text_Count_2 =  KMeans(n_clusters=2)
labels_for_2 = kmeans_text_Count_2.fit(df_text_scaled).labels_
result_2cluster = pd.DataFrame([DF['name'],labels_for_2]).T
kmeans_text_Count_4 =  KMeans(n_clusters=4)
labels_for_4 =kmeans_text_Count_4.fit(df_text_scaled).labels_
result_4cluster = pd.DataFrame([DF['name'],labels_for_4]).T

#distance matrix 
#cos dis
cosdis_text = 1 - cosine_similarity(df_text_scaled)
#eu dis
eudist_text = euclidean_distances(df_text_scaled)
#man dis
mandist_text = manhattan_distances(df_text_scaled)
#2d plot
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
pos_cos = mds.fit_transform(cosdis_text)
pos_eu = mds.fit_transform(eudist_text)
pos_man = mds.fit_transform(mandist_text)
#cos plot 
xs_cos, ys_cos = pos_cos[:,0],pos_cos[:,1]
for x,y,name in zip(xs_cos,ys_cos,DF['name']):
    plt.scatter(x,y)
    plt.text(x,y,name)
# plt.savefig('2d_cos_text.png')
plt.show()
#eu plot
xs_cos, ys_cos = pos_eu[:,0],pos_eu[:,1]
for x,y,name in zip(xs_eu,ys_eu,DF['name']):
    plt.scatter(x,y)
    plt.text(x,y,name)
# plt.savefig('2d_eu_text.png')
plt.show()
#man plot 
xs_cos, ys_cos = pos_man[:,0],pos_man[:,1]
for x,y,name in zip(xs_man,ys_man,DF['name']):
    plt.scatter(x,y)
    plt.text(x,y,name)
# plt.savefig('2d_man_text.png')
plt.show()



#3d part 
#3d
mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
pos_cos = mds.fit_transform(cosdis_text)
pos_eu = mds.fit_transform(eudist_text)
pos_man = mds.fit_transform(mandist_text)
# 3d base on cos dist
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for x, y, z, s in zip(pos_cos[:, 0], pos_cos[:, 1], pos_cos[:, 2], DF['name']):
    plt.scatter(x,y,z)
    ax.text(x, y, z, s)

# ax.set_xlim3d(-.2,.3) #stretch out the x axis
# ax.set_ylim3d(-.2,.3) #stretch out the x axis
# ax.set_zlim3d(-.2,.3) #stretch out the z axis
# plt.savefig('3d_cos_text.png')
plt.show()
# # 3d base on euclidean distance
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos_eu[:, 0], pos_eu[:, 1], pos_eu[:, 2])
for x, y, z, s in zip(pos_eu[:, 0], pos_eu[:, 1], pos_eu[:, 2], DF['name']):
    plt.scatter(x,y,z)
    ax.text(x, y, z, s)

# ax.set_xlim3d(-.2,.3) #stretch out the x axis
# ax.set_ylim3d(-.2,.3) #stretch out the x axis
# ax.set_zlim3d(-.2,.3) #stretch out the z axis
# plt.savefig('3d_eu_text.png')
plt.show()
# #3d base on manhattan distance
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos_man[:, 0], pos_man[:, 1], pos_man[:, 2])
for x, y, z, s in zip(pos_man[:, 0], pos_man[:, 1], pos_man[:, 2], DF['name']):
    plt.scatter(x,y,z)
    ax.text(x, y, z, s)

# ax.set_xlim3d(-.2,.3) #stretch out the x axis
# ax.set_ylim3d(-.2,.3) #stretch out the x axis
# ax.set_zlim3d(-.2,.3) #stretch out the z axis
# plt.savefig('3d_man_text.png')
plt.show()



#ward clustering using cosdist
linkage_matrix_cos = ward(cosdis_text)   #i don't know why here the len(linkage_matrix) is 150 not 151
fig, ax = plt.subplots(figsize=(15, 20))
dendrogram(linkage_matrix_cos)
# plt.savefig('ward_text_cos.png')
plt.show()
#ward clustering using euclidean distance
linkage_matrix_eu = ward(eudist_text)   #i don't know why here the len(linkage_matrix) is 150 not 151
fig, ax = plt.subplots(figsize=(15, 20))
dendrogram(linkage_matrix_eu)
# plt.savefig('ward_text_eu.png')
plt.show()
#ward clustering using manhattan distance
linkage_matrix_man = ward(mandist_text)   #i don't know why here the len(linkage_matrix) is 150 not 151
fig, ax = plt.subplots(figsize=(15, 20))
dendrogram(linkage_matrix_man)
# plt.savefig('ward_text_man.png')
plt.show()



