import Robogame as rg
import networkx as nx
import altair as alt
import time, json
import pandas as pd
import numpy as np
import streamlit as st

import nxviz as nv
from nxviz import annotate
import math

import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
#import statsmodels.api as sm

st.set_option('deprecation.showPyplotGlobalUse', False)

col1, col2 = st.columns([3, 1])

timeVis = st.sidebar.empty()
predVis = st.sidebar.empty()



with col1:
    networkVis = st.empty()
    treeVis = st.empty()
    dataFrameVis = st.empty()
    dataFrameVis2 = st.empty()
with col2:
    dataVis = st.empty()
    

def setNetworkId(socialnet, node_cnt):
    nodeDict = {}
    for id in range(node_cnt):
        nodeDict[id] = id
    nx.set_node_attributes(socialnet, nodeDict, "id")

def setNetworkPopularity(socialnet):
    nx.set_node_attributes(socialnet, nx.degree_centrality(socialnet), "popularity")    

def setExpireNodeSize(expireTime):
    if expireTime <= 0:
        return 0.1
    elif expireTime >= 5:
        return 0.3
    else:
        return (6 - math.ceil(expireTime)) * 0.3

def setNetworkExpire(socialnet, node_cnt, robots, curTime):
    expireTimeDict = {}
    for id in range(node_cnt):
        expireTime = robots.iloc[id]["expires"] - curTime
        expireTimeDict[id] = setExpireNodeSize(expireTime)
    
    nx.set_node_attributes(socialnet, expireTimeDict, "expires")

def setNetworkStatus(socialnet, node_cnt, robots):
    nx.set_node_attributes(socialnet, robots[:node_cnt]["winner"].to_dict(), "status")

def setNetworkTarget(socialnet, node_cnt):
    targetDict = {}
    for id in range(node_cnt):
        targetDict[id] = 1.0
    
    nx.set_node_attributes(socialnet, targetDict, "target")

def drawNetwork(socialnet):
    c = nv.CircosPlot(socialnet, node_color="status", node_order="popularity", node_size="expires")  # node_alpha="target"
    c.fig.set_size_inches(15, 15)
    # nv.circos(socialnet, node_color_by="status", sort_by="popularity", node_size_by="expires", node_alpha_by="target")
    annotate.circos_labels(socialnet, sort_by="popularity")

def initNetwork(socialnet, node_cnt, robots, curTime):
    setNetworkId(socialnet, node_cnt)
    setNetworkPopularity(socialnet)

    setNetworkExpire(socialnet, node_cnt, robots, curTime)
    setNetworkStatus(socialnet, node_cnt, robots)
    setNetworkTarget(socialnet, node_cnt)

    drawNetwork(socialnet)

def updateNetwork(socialnet, node_cnt, robots, curTime):
    setNetworkExpire(socialnet, node_cnt, robots, curTime)
    setNetworkStatus(socialnet, node_cnt, robots)
    setNetworkTarget(socialnet, node_cnt)

    drawNetwork(socialnet)

def getFamily(tree_graph, robot_number):
    # find parent, siblings, children id
    # TODO: encode family with different colors
    parent = list(nx.edge_dfs(tree_graph, robot_number, orientation='reverse'))[0][0]
    siblings = [v for k, v in nx.edge_dfs(tree_graph, parent) if k == parent]
    children = [v for k, v in nx.edge_dfs(tree_graph, robot_number) if k == robot_number]
    return {"parent": parent, "siblings": siblings, "children": children}

def drawTree(G, robot_number):
    # plot family tree of target robot
    G_select = nx.DiGraph()
    family = getFamily(G, robot_number)
    for node in family["siblings"]:
        G_select.add_edge(family["parent"], node)
    for node in family['children']:
        G_select.add_edge(robot_number, node)

    pos_select = graphviz_layout(G_select, prog="dot")
    rotated_pos_select = {node: (-y,x) for (node, (x,y)) in pos_select.items()}

    nx.draw(
        G_select,
        rotated_pos_select,
        with_labels=True,
        node_size=5000,
        font_size=14,
        node_color=["red" if v == robot_number else "#1f78b4" for v in G_select.nodes()]
    )

# def updatePredModel(parthints_df, robots):
#     parthints_df = parthints_df.drop_duplicates('id', keep='last')
#     parthints_df_wide = parthints_df.pivot(index ='id', columns='column', values='value')
#     #parthints_df_wide.sample(10)

#     # link predhint data with robot data
#     new_data = pd.merge(robots, parthints_df_wide, left_on='id', right_on='id', how='left')
#     print(new_data.columns)
#     #Replace non-numric values with NA
#     def isnumber(x):
#         try:
#             float(x)
#             return True
#         except:
#             return False

#     new_data = new_data[new_data.applymap(isnumber)]

#     #Drop the rows where dependent variable are missing.
#     test_data = new_data.loc[new_data['Productivity'].isna()]
#     new_data = new_data.dropna(subset=['Productivity'])

#     #Due to large amount of missing dta, we decied to use those variables with numeric values only, and replace NAN with 0
#     new_data = new_data.fillna(0)

#     x = new_data[['Astrogation Buffer Length','AutoTerrain Tread Count',
#               'Cranial Uplink Bandwidth', 'InfoCore Size',
#               'Polarity Sinks', 'Repulsorlift Motor HP',
#               'Sonoreceptors']]
#     y = new_data['Productivity']

#     # Due to the limited amount of data, we decided not to do train test split
#     # Splitting the varaibles as training and testing
#     # X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.7,
#     #                                                     test_size = 0.3, random_state = 100)

#     # Fitting the resgression line using 'OLS'
#     x = sm.add_constant(x)
#     model = sm.OLS(y.astype(float), x.astype(float)).fit()

#     x_test = test_data[['Astrogation Buffer Length','AutoTerrain Tread Count',
#             'Cranial Uplink Bandwidth', 'InfoCore Size',
#             'Polarity Sinks', 'Repulsorlift Motor HP',
#             'Sonoreceptors']]

#     y_pred = model.predict(x_test)
#     robots['pred_productivity'] = y_pred

#     return robots

if 'key' not in st.session_state:
    game = rg.Robogame("bob")
    game.setReady()
    st.session_state['key'] = game
    #robot_info = game.getRobotInfo()

    game.setBets({i : 50 for i in range(100)})
else:
    game = st.session_state.key
    
    

while(True):
    gametime = game.getGameTime()
    
    timetogo = gametime['gamestarttime_secs'] - gametime['servertime_secs']
    
    if ('Error' in gametime):
        print("Error"+str(gametime))
        break
    if (timetogo <= 0):
        print("Let's go!")
        break
        
    print("waiting to launch... game will start in " + str(int(timetogo)))
    time.sleep(1) # sleep 1 second at a time, wait for the game to start

# get robot info 
robot_info = game.getRobotInfo()
robot_number = predVis.selectbox('Enter a robot number', options=robot_info['id'].tolist())
dataFrameVis2.write(robot_info)

# add form submission
guess = st.sidebar.text_input('Enter your guess:', max_chars=3)
with st.sidebar.form(key='my_form'):
    submit = st.form_submit_button('submit')
    if submit and guess != '' and robot_number:
        game.setBets({robot_number : int(guess)})



# get family tree
tree = game.getTree()
G = nx.tree_graph(tree)

# get network
network = game.getNetwork()
socialnet = nx.node_link_graph(network)


node_cnt = len(network['nodes'])
curTime = game.getGameTime()['curtime']

networkVis.pyplot(initNetwork(socialnet, node_cnt, robot_info, curTime))

while True:
    
    
    game.getHints()
    predHints = game.getAllPredictionHints()
    # partHints = game.getAllPartHints()
	
    predhints_df = pd.read_json(json.dumps(predHints),orient='records')
    # parthints_df = pd.read_json(json.dumps(partHints),orient='records')
    # model = updatePredModel(parthints_df, robot_info)

    # y_pred = model.predict(robot_info)

    gametime = game.getGameTime()
    timeVis.write(round((gametime['servertime_secs'] - gametime['gamestarttime_secs'])/6, 2))


    # get all time
    timeset = list(set(predhints_df["time"].tolist()))
    pointset = []
    for i in timeset:
        pointset.append(predhints_df[predhints_df["time"]==i]["value"].mean())



    fit = np.polyfit(timeset, pointset, 3)
    p = np.poly1d(fit)
    result_points = [p(i) for i in np.arange(0, 100)]
    fity = []
    for x in result_points:

        if (x > 100):  # we know y can't be > 100
            x = 100
        if (x < 0):  # we know y can't be < 0
            x = 0
        fity.append(x)


    prediction_df = pd.DataFrame(data={'time' : np.arange(0, 100).tolist(), 'pred' : fity, 'type': ['BasedOnAverage' for i in range(0, 100)]})

    predhints_df.drop_duplicates(inplace=True)

    dataFrameVis.dataframe(predhints_df)

    df_robot_number = predhints_df[predhints_df['id'] ==robot_number]


    timeset2 = list(set(df_robot_number["time"].tolist()))

    if len(timeset2) > 1:
        pointset2 = []
        for i in timeset2:
            pointset2.append(df_robot_number[df_robot_number["time"]==i]["value"].mean())


        fit2 = np.polyfit(timeset2, pointset2, 3)
        p2 = np.poly1d(fit2)
        result_points2 = [p2(i) for i in np.arange(0, 100)]
        fity2 = []
        for x in result_points2:

            if (x > 100):  # we know y can't be > 100
                x = 100
            if (x < 0):  # we know y can't be < 0
                x = 0
            fity2.append(x)
        prediction_df2 = pd.DataFrame(data={'time': np.arange(0, 100).tolist(), 'pred': fity2, 'type': ['BasedOnTheRobot' for i in range(0, 100)]})
        prediction_df = prediction_df.append(prediction_df2)
        predhints_df['id'] = pd.to_numeric(predhints_df['id'])
    selection = alt.selection(type='single', on='mouseover', nearest=True, fields=['time'], empty='none')
    
    
    slider = alt.binding_range(min=0, max=100, step=1, name='cutoff:')
    selection_for_robot_number = alt.selection_single(name="SelectorName", fields=['cutoff'],
                                bind=slider, init={'cutoff': robot_number})
    
    
    # draw hints from hacker
    base = alt.Chart(predhints_df) 
    
    
    # draw prediction orange line
    base2 = alt.Chart(prediction_df)

    # fit line based on some point know about the robot

   # tooltip=['id', 'time', 'value']

    c1 = base.mark_point().encode(
		x='time:Q',
		y='value:Q',
        color=alt.condition(
            alt.datum.id == selection_for_robot_number.cutoff,
            alt.value('red'), alt.value('#1f77b4')
        )
    ).add_selection(
        selection_for_robot_number
    )
    
    
    # pot 
    
    rules = base2.mark_rule(color='red', strokeDash=[1,1]).encode(
        x="time:Q"
    ).transform_filter(
        selection
    )

    selectors = base2.mark_point().encode(
        x='time:Q',
        opacity=alt.value(0),
    ).add_selection(
        selection
    )



    ## ployfit numpy of prediction
    c3 = base2.mark_line().encode(
        x='time:Q',
        y='pred:Q',
        color='type:N'
    )

    # add text over the verticle line
    text2 = c3.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(selection, 'pred:Q', alt.value(' '))
    )



    # polt invisable points make the line can be select
    
    c4 = base2.mark_point(color='red', opacity=0).encode(
        x='time:Q',
        y='pred:Q',
    )


    # plot for specific robot, df_robot_number was created based on the selection of robot
    # Simply connect them.

    c5 = alt.Chart(df_robot_number).mark_line(color='orange', opacity=0.7).encode(
        x='time:Q',
        y='value:Q',
    )

    # add red verticle line for the graph
    
    expire = alt.Chart(pd.DataFrame({
        'time': robot_info[robot_info['id']==robot_number]['expires'].tolist(),
    })).mark_rule(color='red').encode(
        x='time:Q'
    )
    
    # final concatented graph
    result2 = alt.layer(
       c3, c4, rules, selectors, text2, c1, c5, expire
    ).interactive(bind_y=False, ).properties(
    width=500, height=500
    )

    dataVis.altair_chart(result2)
    treeVis.pyplot(drawTree(G, robot_number))
    time.sleep(3)