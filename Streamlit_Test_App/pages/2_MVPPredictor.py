import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError
import pandas as pd
import numpy as np
import requests as rq
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
import time
import requests
import shutil
import os.path
from unidecode import unidecode


st.set_page_config(page_title="MVP Predictor")

st.markdown("""
<style>
button {
    height: 50px;
    width: 200px;
    color: blue;
}
</style>
""", unsafe_allow_html=True)

st.title("Major League Baseball MVP Predictor")
st.markdown("The Major League Baseball Most Valuable Player Award is an annual Major League Baseball award given to one outstanding player in the American League and one in the National League. The award has been presented by the Baseball Writers' Association of America since 1931. The predictor below relies on batting data dating back to the 1960 MLB season to predict the winners of the American and National league Most Valuable Players for every season since 2010.")
st.sidebar.header("MVP Predictor")

years = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])

year = st.selectbox("Choose a year", years, index=None, placeholder='No year selected')

def make_prediction(year):
    stats = pd.read_csv('Batting.csv')
    stats["yearID"] = stats["yearID"].astype(int)
    stats = stats[stats['yearID'] > 1960]
    awards = pd.read_csv('AwardsSharePlayers.csv')
    awards['yearID'] = awards['yearID'].astype(int)
    awards = awards[awards['awardID'] == 'MVP']
    awards = awards[awards['yearID'] > 1960]
    merged = pd.merge(stats, awards, on=['yearID','playerID'])
    ## plate appearances are the total of at-bats, walks, hits-by-pitch, and sacrifice flies
    merged['PA'] = merged['AB'] + merged['BB'] + merged['HBP'] + merged['SF']
    ## pointsProp stores the proportion of the maximum number of points that the player scored
    merged['pointsProp'] = merged['pointsWon'] / merged['pointsMax']
    ## slugging percentage is equal to the total number of bases a player gets per at-bat
    merged['SLG'] = ((merged['H'] - merged['2B'] - merged['3B'] - merged['HR']) + 2*merged['2B'] + 3*merged['3B'] + 4*merged['HR']) / merged['AB']
    ## on-base percentage measures how many times a player gets safely on-base per plate appearance
    merged['OBP'] = (merged['H'] + merged['BB'] + merged['HBP']) / merged['PA']
    ## on-base plus slugging is the sum of on-base percentage and slugging percentage
    merged['OPS'] = merged['SLG'] + merged['OBP']
    ## batting average measures the number of hits a player gets per at-bat
    merged['AVG'] = merged['H'] / merged['AB']
    ## all other stats are divided by number of at-bats to control for different players having differing numbers of at-bats
    merged['2B'] = merged['2B'] / merged['AB']
    merged['3B'] = merged['3B'] / merged['AB']
    merged['HR'] = merged['HR'] / merged['AB']
    merged['RBI'] = merged['RBI'] / merged['AB']
    merged['R'] = merged['R'] / merged['AB']
    ## walks are divided by plate appearances, as they count as plate appearances but do not count as at-bats
    merged['BB'] = merged['BB'] / merged['PA']
    merged['SB'] = merged['SB'] / merged['AB']
    merged['SO'] = merged['SO'] / merged['AB']
    extra = merged[['playerID', 'teamID', 'stint', 'lgID_y', 'votesFirst', 'lgID_x', 'awardID',
                'CS', 'IBB', 'HBP', 'SF', 'GIDP', 'pointsMax', 'SH', 'pointsWon', 'yearID', 'pointsProp', 'AB',
                'H', 'G', '3B', '2B', 'PA', 'SB']].copy()
    merged = merged.drop(['playerID', 'teamID', 'stint', 'lgID_y', 'votesFirst', 'lgID_x', 'awardID',
                'CS', 'IBB', 'HBP', 'SF', 'GIDP', 'pointsMax', 'SH', 'pointsWon', 
                        'yearID', 'AB', 'H', 'G', '3B', '2B', 'PA', 'SB'], axis=1)
    merged = merged.dropna(axis=0)
    ## labels stores the points proportion, the value that the random forest will try to predict
    labels = np.array(merged['pointsProp'])
    ## pointsProp is now removed from the dataframe so the rest of the columns can become features
    merged = merged.drop('pointsProp', axis = 1)
    merged_list = list(merged.columns)
    ## features stores the information that will be used to predict points proportion
    features = np.array(merged)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(train_features, train_labels)
    y_pred = rf.predict(test_features)
    df=pd.DataFrame({'Actual':test_labels, 'Predicted':y_pred}) 
    predictions = rf.predict(test_features)
    newdata = pd.read_csv('Hitting_Data/mlb-player-stats-Batters{}.csv'.format(year))
    df2 = newdata
    df2['PA'] = df2['AB'] + df2['BB'] + df2['HBP'] + df2['SF']
    df2['2B'] = df2['2B'] / df2['AB']
    df2['3B'] = df2['3B'] / df2['AB']
    df2['HR'] = df2['HR'] / df2['AB']
    df2['R'] = df2['R'] / df2['AB']
    df2['RBI'] = df2['RBI'] / df2['AB']
    df2['BB'] = df2['BB'] / df2['PA']
    df2['SO'] = df2['SO'] / df2['AB']


    df2 = newdata.drop(['Player', 'Team', 'Pos', 'Age', 'CS', 'SH', 'SF', 'HBP', 'AB', 'G', 
                        'H', '3B', '2B', 'PA', 'SB'], axis=1)
    df2 = df2.drop(newdata[newdata['AB'] < 450].index)
    df2 = np.array(df2)
    newpred = rf.predict(df2)
    newpredsorted = newpred
    newpredsorted = sorted(newpredsorted, reverse=True)
    ind = []
    for i in newpredsorted[:200]:
        ind.append(np.where(newpred == i))

    indices = []
    for i in ind[:100]:
        word = str(i)
        word = word.split('[')[1]
        word = word.split(']')[0]
        indices.append(int(word))

    

    ## two arrays are created to store the players from the American and National Leagues
    alplayers = []
    nlplayers = []

    ## Two arrays that store the teams that play in each league
    AL = ['OAK', 'SEA', 'LAA', 'HOU', 'TEX', 'MIN', 'DET', 'NYY', 'TB', 'KC', 'CWS', 'BOS', 'BAL', 'CLE', 'TOR']
    NL = ['SF', 'LAD', 'SD', 'ARI', 'NYM', 'MIA', 'MIL', 'ATL', 'WSH', 'CIN', 'PHI', 'STL', 'CHC', 'COL', 'PIT']

    ## Loop that adds players to either alplayers or nlplayers based on which league their team belongs to
    for i in indices:
        if newdata.iloc[i]['Team'] in AL:
            alplayers.append([newdata.iloc[i]['Player'], newdata.iloc[i]['Team']])
        else:
            nlplayers.append([newdata.iloc[i]['Player'], newdata.iloc[i]['Team']])
    return alplayers, nlplayers

st.image("media/yankee_stadium.jpg")
if st.button("Predict"):
    with st.spinner(text='Making prediction...'):
        ind = make_prediction(year)
    almvp = str(ind[0][0][0]).split(" ")
    nlmvp = str(ind[1][0][0]).split(" ")
    for i in range(2):
        almvp[i] = str(unidecode(almvp[i]))            
        nlmvp[i] = str(unidecode(nlmvp[i]))
    
    html_page_al = requests.get("https://en.wikipedia.org/wiki/" + almvp[0] + "_" + almvp[1])
    html_page_nl = requests.get("https://en.wikipedia.org/wiki/" + nlmvp[0] + "_" + nlmvp[1])
    print("al: https://en.wikipedia.org/wiki/" + almvp[0] + "_" + almvp[1])
    print("nl html: https://en.wikipedia.org/wiki/" + nlmvp[0] + "_" + nlmvp[1])
    soup_al = bs(html_page_al.content, 'html.parser')
    soup_nl = bs(html_page_nl.content, 'html.parser')
    book_container_al = soup_al.find('a', class_='mw-file-description')
    book_container_nl = soup_nl.find('a', class_='mw-file-description')
    print("bca1:", book_container_al)
    print("bcn1:", book_container_nl)
    
    print("book al:", book_container_al)
    print("book nl:", book_container_nl)



    

    image_al = book_container_al.findAll('img', class_="mw-file-element")[0].attrs['src']

    image_nl = book_container_nl.findAll('img', class_="mw-file-element")[0].attrs['src']
    image_al = "https:" + image_al
    image_nl = "https:" + image_nl
    print("image_al:", image_al)
    print("image nl:", image_nl)

    print("Url:", str(image_al))
    print("Url:", str(image_nl))

    r_al = requests.get(str(image_al), stream=True)
    r_nl = requests.get(str(image_nl), stream=True)

    

    try:
        if r_al.status_code == 200:  # Check for successful response
            filename = "media/" + almvp[0] + "_" + almvp[1] + ".jpg"
            with open(filename, 'wb') as f:
                r_al.raw.decode_content = True
                shutil.copyfileobj(r_al.raw, f)
            print(f"File saved successfully to {filename}")
        else:
            print(f"Failed to download file. Status code: {r_al.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        if r_nl.status_code == 200:  # Check for successful response
            filename = "media/" + nlmvp[0] + "_" + nlmvp[1] + ".jpg"
            with open(filename, 'wb') as f:
                r_nl.raw.decode_content = True
                shutil.copyfileobj(r_nl.raw, f)
            print(f"File saved successfully to {filename}")
        else:
            print(f"Failed to download file. Status code: {r_nl.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")
    

    al_winners = [[2010, "Josh Hamilton (TEX)"], [2011, "Justin Verlander (DET)"], [2012, "Miguel Cabrera (DET)"], [2013, "Miguel Cabrera (DET)"], [2014, "Mike Trout (LAA)"], [2015, "Josh Donaldson (TOR)"], [2016, "Mike Trout (LAA)"], [2017, "Jose Altuve (HOU)"], [2018, "Mookie Betts (BOS)"], [2019, "Mike Trout (LAA)"], [2020, "Jose Abreu (CWS)"], [2021, "Shohei Ohtani (LAA)"], [2022, "Aaron Judge (NYY)"], [2023, "Shohei Ohtani (LAA)"]]
    nl_winners = [[2010, "Joey Votto (CIN)"], [2011, "Ryan Braun (MIL)"], [2012, "Buster Posey (SFG)"], [2013, "Andrew McCutchen (PIT)"], [2014, "Clayton Kershaw (LAD)"], [2015, "Bryce Harper (WAS)"], [2016, "Kris Bryant (CHC)"], [2017, "Giancarlo Stanton (MIA)"], [2018, "Christian Yelich (MIL)"], [2019, "Cody Bellinger (LAD))"], [2020, "Freddie Freeman (ATL)"], [2021, "Bryce Harper (PHI)"], [2022, "Paul Goldschmidt (STL)"], [2023, "Ronald Acuna (ATL)"]]
    st.success("Done!")
    st.header("The projected {} American League Most Valuable Player is *{}* ({}).".format(year, str(ind[0][0][0]), str(ind[0][0][1])), divider="red")
    st.header("The projected {} National League Most Valuable Player is *{}* ({}).".format(year, str(ind[1][0][0]), str(ind[1][0][1])), divider="blue")

    col1, col2 = st.columns(2)
    with col1:
        st.image("media/" + almvp[0] + "_" + almvp[1] + ".jpg", width=150, caption=("American League predicted winner: " + almvp[0] + " " + almvp[1]))
    with col2:
        st.image("media/" + nlmvp[0] + "_" + nlmvp[1] + ".jpg", width=150, caption=("National League predicted winner: " + nlmvp[0] + " " + nlmvp[1]))
    st.write("###")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("\n\nProjected American League Rankings\n", divider='red')
        for i in range(5):
            st.write(i+1, (str(ind[0][i][0])) + " (" + (str(ind[0][i][1])) + ")" + "\n")
        if(year < 2024):
            st.subheader("Actual {} American League MVP Winner".format(year), divider='red')
            winner = "#### "
            winner += al_winners[year-2010][1]
            st.markdown(winner)
    with col2:
        st.subheader("\nProjected National League Rankings\n", divider='blue')
        for i in range(5):
            st.write(i+1, (str(ind[1][i][0])) + " (" + (str(ind[1][i][1])) + ")" + "\n")
        if(year < 2024):
            st.subheader("Actual {} National League MVP Winner".format(year), divider='blue')
            winner = "#### "
            winner += nl_winners[year-2010][1]
            st.markdown(winner)

    if year < 2024:
        projected_al = (str(ind[0][0][0])) + " (" + (str(ind[0][0][1])) + ")"
        projected_nl = (str(ind[1][0][0])) + " (" + (str(ind[1][0][1])) + ")"
        actual_al = str(al_winners[year-2010][1])
        actual_nl = str(nl_winners[year-2010][1])

        print("proj al:", projected_al)
        print("proj nl:", projected_nl)
        print("act al:", actual_al)
        print("act nl:", actual_nl)
        
        if projected_al != actual_al and projected_nl != actual_nl:
            st.error("Neither award was predicted correctly")
        elif projected_al == actual_al and projected_nl != actual_nl:
            st.warning("The American League winner was predicted correctly but the National League winner was not")
        elif projected_al != actual_al and projected_nl == actual_nl:
            st.warning("The National League winner was predicted correctly but the American League winner was not")
        else:
            st.success("Both awards were predicted correctly")
    else:
        st.warning("The Most Valuable Player Awards have not been given yet for {}".format(year))