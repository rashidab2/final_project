import pandas as pd, matplotlib.pyplot as plt, numpy as np
from matplotlib import colors
from datetime import datetime as date 
from sklearn.linear_model import LinearRegression


#gets the teams agv ratings 
def get_team_avg():
    df = pd.read_csv("nba2k.csv")
    df = df[["full_name","rating","team"]]
    return df.groupby('team').mean(numeric_only=True).rating

#plots the avg rating of all the teams
def plot_rating_avg(df):
    df.plot.bar(rot=0, fontsize=5, color='g')
    plt.title("Team Rating Avg", fontsize=24)
    plt.xlabel("NBA Teams", fontsize=20)
    plt.ylabel("Rating(nba rating)", fontsize=20)
    plt.show()

# geting the age of the player and adding it to the df
def get_players_age():
    age = 0
    df = pd.read_csv("nba2k.csv")
    df = df[["full_name","rating","b_day"]]
    df['Date']= pd.to_datetime(df['b_day'])
    df['Age'] = (date(2019, 9, 6)- df['Date'])
    df['Age'] = df['Age'].astype('<m8[Y]')
    df = df[["full_name","rating",'Age']]
    return df

#histagram of the ratings 
def hist_rating(df):
    x2 = df['rating']
    s = pd.Series(x2)
    bin = range(70, 99, 2)
    ax = s.plot.hist(bins=bin, color='b', ec='black', fontsize=16)
    plt.ylabel('Frequency', fontsize=20, labelpad=10)
    plt.xlabel('Rating(nba rating)', fontsize=20)
    plt.title('NBA2k Rating', fontsize=24, pad=10)
    ax2 = ax.twinx()
    s.plot.hist(ax=ax2, bins=bin, color='b', ec='black', fontsize=16, density=True)
    ax2.set_ylabel('Est Prob Density', fontsize=20, labelpad=10)
    yticks = np.array([0, 4/17, 8/17, 12/17,16/17,1])
    yticklabels = ['0', '4/17', '8/17', '12/17','16/17','1']
    plt.yticks(yticks, yticklabels)
    plt.show()

#create a new series with the top 5 teams 
def top_5():
    df = get_team_avg()
    s = df.nlargest(5)
    return s

#gets the players of the top 5 
#chooses players to go against eachother
def starting_5():
    s = top_5()
    df = pd.read_csv("nba2k.csv")
    df = df[['Nets','Warriors','Clippers','Lakers','Blazers']]
    df = df.head(5)
    data = [['W','C','L','N'],
            ['W','Tie','L','W'],
            ['C','Tie','L','C'],
            ['L','L','L','L'],
            ['N','W','C','L']]
    index = ['Nets','Warriors','Clippers','Lakers','Blazers']
    columns = ['Round1','Round2','Round3','Round4']
    df2 = pd.DataFrame(data, index,columns)
    data2 = ['Warriors','Clippers','Lakers','Nets','Warriors','Tie(W/C)','Lakers','Warriors','Clippers','Tie(W/C)',
              'Lakers','Clippers','Lakers','Lakers','Lakers','Lakers','Nets','Warriors','Clippers','Lakers']
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    df = pd.DataFrame({'Winnings': data2})
    df['Winnings'].value_counts().plot(ax=ax, kind='bar', xlabel='Winning teams', ylabel='frequency')
    ax2 = ax.twinx()
    ax2.set_ylabel('Est Prob', fontsize=20, labelpad=10)
    yticks = np.array([0,1/4,2/4,3/4,1])
    yticklabels = ['0','1/4','2/4','3/4','1']
    plt.yticks(yticks, yticklabels)
    plt.title('Top 5 Teams Winnings', fontsize=24, pad=10)
    plt.show()
    return df,df2

#scatter Plot of players age and rating
def scatter_plt(df):
    s = df.groupby('Age').mean(numeric_only=True).rating
    df = s.to_frame()
    df.reset_index(inplace=True)
    df.columns = ['Age','Rating']
    #df.plot(kind='scatter', x='Age',y='Rating')
    plt.title('Age vs Rating')
    plt.xlabel('Age(yrs old)', fontsize=15, labelpad=10)
    plt.ylabel('Rating(nba rating)', fontsize=15)
    x= df.Age.values.reshape(-1, 1)
    y = df.Rating.values.reshape(-1, 1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(x,y)
    Y_pred = linear_regressor.predict(x)
    plt.scatter(x, y)
    corr = df['Age'].corr(df['Rating'])
    plt.plot(x, Y_pred, color='Black', label = ("corr = "+str(round(corr, 2))))
    plt.legend()
    plt.show()
    return corr

    

def main():
    avg_rating_df = get_team_avg()
    print(avg_rating_df) #every teams average rating
    print(top_5()) #series of top 5 teams
    starting_5() # frequency chart of top 5 team wins vs each other.
    player_age = get_players_age() #return a dataframe of age and rating 
    print(scatter_plt(player_age))
    hist_rating(player_age) #histogram of rating 
    plot_rating_avg(avg_rating_df) #Bar chat of avg rating per team 


if __name__ == '__main__':
    main()