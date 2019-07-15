import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_books = pd.read_csv('books.csv')
df_ratings = pd.read_csv('ratings.csv')

def criteria(i):
    return str(i['authors'])+' '+str(i['original_title'])+' '+str(i['title'])+' '+str(i['language_code'])
df_books['Criteria'] = df_books.apply(criteria,axis='columns')
# print(df_books.head())

# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(tokenizer=lambda x:x.split(' '))
matrixFeature = model.fit_transform(df_books['Criteria'])

kriteria = model.get_feature_names()
jml_fitur = len(kriteria)

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
skor = cosine_similarity(matrixFeature)

# menginput buku yang ratingnya diatas 3 star
andi1 = df_books[df_books['original_title']=='The Hunger Games']['book_id'].tolist()[0]-1 
andi2 = df_books[df_books['original_title']=='Catching Fire']['book_id'].tolist()[0]-1 
andi3 = df_books[df_books['original_title']=='Mockingjay']['book_id'].tolist()[0]-1 
andi4 = df_books[df_books['original_title']=='The Hobbit or There and Back Again']['book_id'].tolist()[0]-1 
sukaAndi = [andi1,andi2,andi3,andi4]

budi1 = df_books[df_books['original_title']=='Harry Potter and the Philosopher\'s Stone']['book_id'].tolist()[0]-1 
budi2 = df_books[df_books['original_title']=='Harry Potter and the Chamber of Secrets']['book_id'].tolist()[0]-1 
budi3 = df_books[df_books['original_title']=='Harry Potter and the Prisoner of Azkaban']['book_id'].tolist()[0]-1 
sukaBudi = [budi1,budi2,budi3]

ciko1 = df_books[df_books['original_title']=='Robots and Empire']['book_id'].tolist()[0]-1 
sukaCiko = [ciko1]

dedi1 = df_books[df_books['original_title']=='Nine Parts of Desire: The Hidden World of Islamic Women']['book_id'].tolist()[0]-1 
dedi2 = df_books[df_books['original_title']=='A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam']['book_id'].tolist()[0]-1 
dedi3 = df_books[df_books['original_title']=='No god but God: The Origins, Evolution, and Future of Islam']['book_id'].tolist()[0]-1 
sukaDedi = [dedi1,dedi2,dedi3]

ello1 = df_books[df_books['original_title']=='Doctor Sleep']['book_id'].tolist()[0]-1 
ello2 = df_books[df_books['original_title']=='The Story of Doctor Dolittle']['book_id'].tolist()[0]-1 
ello3 = df_books[df_books['title']=='Bridget Jones\'s Diary (Bridget Jones, #1)']['book_id'].tolist()[0]-1 
sukaEllo = [ello1,ello2,ello3]

skorAndi1 = list(enumerate(skor[andi1]))
skorAndi2 = list(enumerate(skor[andi2]))
skorAndi3 = list(enumerate(skor[andi3]))
skorAndi4 = list(enumerate(skor[andi4]))

skorBudi1 = list(enumerate(skor[budi1]))
skorBudi2 = list(enumerate(skor[budi2]))
skorBudi3 = list(enumerate(skor[budi3]))

skorCiko = list(enumerate(skor[ciko1]))

skorDedi1 = list(enumerate(skor[dedi1]))
skorDedi2 = list(enumerate(skor[dedi2]))
skorDedi3 = list(enumerate(skor[dedi3]))

skorEllo1 = list(enumerate(skor[ello1]))
skorEllo2 = list(enumerate(skor[ello2]))
skorEllo3 = list(enumerate(skor[ello3]))

skorAndi = []
for i in skorAndi1:
    skorAndi.append((i[0],0.25*(skorAndi1[i[0]][1]+skorAndi2[i[0]][1]+skorAndi3[i[0]][1]+skorAndi4[i[0]][1])))
skorBudi = []
for i in skorAndi1:
    skorBudi.append((i[0],(skorBudi1[i[0]][1]+skorBudi2[i[0]][1]+skorBudi3[i[0]][1])/3))
skorDedi = []
for i in skorAndi1:
    skorDedi.append((i[0],(skorDedi1[i[0]][1]+skorDedi2[i[0]][1]+skorDedi3[i[0]][1])/3))
skorEllo = []
for i in skorAndi1:
    skorEllo.append((i[0],(skorEllo1[i[0]][1]+skorEllo2[i[0]][1]+skorEllo3[i[0]][1])/3))

sort_andi = sorted(skorAndi, key=lambda j:j[1], reverse=True)
sort_budi = sorted(skorBudi, key = lambda j:j[1], reverse = True)
sort_ciko = sorted(skorCiko, key = lambda j:j[1], reverse = True)
sort_dedi = sorted(skorDedi, key = lambda j:j[1], reverse = True)
sort_ello = sorted(skorEllo, key = lambda j:j[1], reverse = True)

# top 5 recommendation
sama_andi = []
for i in sort_andi:
    if i[1]>0:
        sama_andi.append(i)
sama_budi = []
for i in sort_budi:
    if i[1]>0:
        sama_budi.append(i)
sama_ciko = []
for i in sort_ciko:
    if i[1]>0:
        sama_ciko.append(i)
sama_dedi = []
for i in sort_dedi:
    if i[1]>0:
        sama_dedi.append(i)
sama_ello = []
for i in sort_ello:
    if i[1]>0:
        sama_ello.append(i)

print('1. Buku bagus untuk Andi:')
for i in range(0,5):
    if sama_andi[i][0] not in sukaAndi:
        print('-',df_books['original_title'].iloc[sama_andi[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[sama_andi[i][0]])

print(' ')
print('2. Buku bagus untuk Budi:')
for i in range(0,5):
    if sama_budi[i][0] not in sukaBudi:
        print('-',df_books['original_title'].iloc[sama_budi[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[sama_budi[i][0]])

print(' ')
print('3. Buku bagus untuk Ciko:')
for i in range(0,5):
    if sama_ciko[i][0] not in sukaCiko:
        print('-',df_books['original_title'].iloc[sama_ciko[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[sama_ciko[i][0]])

print(' ')
print('4. Buku bagus untuk Dedi:')
for i in range(0,5):
    if sama_dedi[i][0] not in sukaDedi:
        print('-',df_books['original_title'].iloc[sama_dedi[i][0]])
    else:
        i+=5
        print('-',df_books['original_title'].iloc[sama_dedi[i][0]])

print(' ')
print('5. Buku bagus untuk Ello:')
for i in range(0,5):
    if sama_ello[i][0] not in sukaEllo:
        if str(df_books['original_title'].iloc[sama_ello[i][0]])=='nan':
            print('-',df_books['title'].iloc[sama_ello[i][0]])
        else:
            print('-',df_books['original_title'].iloc[sama_ello[i][0]])  
    else:
        i+=5
        if str(df_books['original_title'].iloc[sama_ello[i][0]])=='nan':
            print('-',df_books['title'].iloc[sama_ello[i][0]])
        else:
            print('-',df_books['original_title'].iloc[sama_ello[i][0]])