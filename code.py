import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

veri = pd.read_csv("data.csv")
# print(veri)

# Soru1-a)

s1 = veri.sort_values("Score", ascending=False)
s2 = s1.iloc[0:10, :]
s3 = s2.drop("Metascore", axis=1)
s4 = s3.dropna()
s4 = s4.groupby("Title")["Revenue"].sum()

x = s4.index.tolist()
y = s4.values.tolist()
# print(x)
# print(y)

plt.pie(y, labels=x, autopct='%1.0f%%', shadow=True, startangle=90, pctdistance=0.6, textprops={'fontsize': 10})
# plt.show()

# Soru1-b)

s5 = s1.iloc[0:30, :]
s5 = s5.drop("Metascore", axis=1)
s6 = s5.dropna()

plt.figure(figsize=(25, 25))
sns.barplot(x=s6.Revenue, y=s6.Title)
plt.xlabel("Hasılat", size=10)
plt.ylabel("Film Adı", size=10)
plt.title("Film Hasılatları", size=20)
plt.yticks(size=10)
plt.xticks(size=10)
# plt.figure()
# plt.show()

# Soru2

s7 = veri[(veri["Genre"] == "Action") | (veri["Genre"] == "Drama") | (veri["Genre"] == "Comedy")]
s7 = s7[s7["Year"] > 2010]
s7["Title"] = 1
grup = s7.groupby(["Year", "Genre"])["Title"].sum()
s8 = grup.copy().reset_index()
sns.lineplot("Year", s8.Title, hue="Genre", data=s8, linewidth=2.5, style="Genre", markers=True, dashes=False)

plt.xlabel("YÄ±llar", size=12)
plt.ylabel("Toplam Film SayÄ±sÄ±", size=12)
plt.title("Film tÃ¼rlerinin yÄ±llara gÃ¶re sayÄ±sÄ±", size=15)
# plt.show()

# Soru3

veri2 = veri.copy()
veri2["Title"] = 1
grup1 = veri2.groupby("Director")["Title"].sum()
grup1 = grup1.copy().reset_index().sort_values("Title", ascending=False).iloc[0:1, 0].iloc[0]

sart1 = veri["Director"] == grup1
veri3 = veri[(sart1)]
Soru3 = veri3[veri3["Revenue"] == max(veri3["Revenue"])]


# print(Soru3)

# Soru4

def f(Description):
    if "hacker" in Description.lower():
        return True
    return False


Soru4 = veri[veri["Description"].apply(f)].sort_values("Score", ascending=False)
# print(Soru4)

#Soru5

sart1=veri["Year"]>2012
sart2=veri["Year"]<2016
veri2=veri[(sart1) & (sart2)]


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
veri2["Year"]= le.fit_transform(veri2["Year"])
k = veri2["Year"].max()
for j, i in enumerate(range(0, k+1)):
    j = j+2013
    sonuc = veri2[(veri2["Year"] == i)]
    sonuc2 = sonuc[sonuc["Score"] == sonuc["Score"].max()]["Title"].iloc[0]
    v1 = str(j)
    print(v1+":"+sonuc2)