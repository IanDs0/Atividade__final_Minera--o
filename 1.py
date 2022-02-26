import numpy as np
import pandas as pd

# lol = pd.read_csv("./2021_LoL_esports.csv")
lol = pd.read_csv("./2021_LoL_esports (cópia).csv")
lol.replace(np.nan, 0, inplace= True)
lol['Firstblood'] = lol['firstblood'].map({1.0:'fb' , 0.0:'no_fb'})
lol['Result'] = lol['result'].map({1:'win' , 0:'lose'})

lol.loc[lol.position=='team',('playername','champion')]='team'
# lol.loc[lol.position==0,('ban1','ban2','ban3','ban4','ban5')]='DC'

# lol['Playername'] = lol['playername'].map({0:'team'})

lol = lol.drop(columns=['patch','gameid','url','datacompleteness','year','date','split','playerid','teamid','opp_csat15','golddiffat15','xpdiffat15','csdiffat15','killsat15',
    'assistsat15','deathsat15','opp_killsat15','opp_assistsat15','opp_deathsat15','assistsat10','deathsat10','opp_killsat10','opp_assistsat10','opp_deathsat10','goldat15',
    'xpat15','csat15','opp_goldat15','opp_xpat15','goldat10','xpat10','csat10','opp_goldat10','opp_xpat10','opp_csat10','golddiffat10','xpdiffat10','csdiffat10','killsat10',
    'monsterkills','monsterkillsownjungle','monsterkillsenemyjungle','teamdeaths','doublekills','triplekills','quadrakills','pentakills','teamkills','gamelength','team kpm',
    'elementaldrakes','opp_elementaldrakes','chemtechs','hextechs','dragons (type unknown)','turretplates','opp_turretplates','ckpm','dpm','damageshare','damagetakenperminute',
    'damagemitigatedperminute','wpm','wcpm','vspm','earned gpm','earnedgoldshare','gspd','cspm',
    
    'damagetochampions','total cs','inhibitors','opp_inhibitors','wardsplaced','wardskilled','controlwardsbought','visionscore','totalgold','earnedgold','goldspent','opp_heralds',
    'firstbaron','barons','opp_barons','firsttower','towers','opp_towers','firstmidtower','firsttothreetowers','minionkills','dragons','opp_dragons','infernals','mountains',
    'clouds','oceans','elders','opp_elders','firstherald','heralds','kills','deaths','assists','firstbloodkill','firstbloodassist','firstbloodvictim','firstdragon',

    'firstblood','result','participantid','playoffs','game','ban1','ban2','ban3','ban4','ban5'
    ])

# lol 0 lol.drop()
# lol.head(12)

lol.to_csv("./lol.csv")
# lol = pd.read_csv("./lol.csv",header = None)
lol.head(10)

def removeAllOcurrencesOfValueInList(_list, value):
    return list(filter(lambda x: x != value, _list))

lista_lol = []

for index, row in lol.iterrows():
    L_lol = row.values.tolist()
    L_lol = removeAllOcurrencesOfValueInList(L_lol,0)
    lista_lol.append(L_lol)

print(lista_lol[0]) 

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(lista_lol).transform(lista_lol)
lol = pd.DataFrame(te_ary, columns=te.columns_)

lol.head(1)

from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(lol, min_support = 0.01, use_colnames = True)
frequent_itemsets.sort_values(by=['support'], ascending = False).head(10)

from mlxtend.frequent_patterns import association_rules

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules.sort_values(by=['lift'], ascending = False).drop(['antecedent support', 'consequent support', 'leverage', 'conviction'], axis=1)


print("Olá mundo")