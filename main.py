import datetime

import numpy as np
import ocpa.algo.predictive_monitoring.factory

from gnn_utils import *
from ocpa.objects.log.importer.csv import factory as csv_import_factory
from ocpa.objects.log.importer.ocel import factory as ocel_import_factory
from ocpa.algo.predictive_monitoring import factory as predictive_monitoring
from ocpa.objects.log.util import misc as log_util
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import pandas as pd
import networkx as nx
from graph_embedding import convert_to_nx_graphs, embed
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

params = {"sap": {"batch_size":4,"lr":0.001,"epochs":15}}

NEXT_ACTIVITY = "next_activity"
def next_activity(node,ocel,params):
    act = params[0]
    e_id = node.event_id
    out_edges = ocel.graph.eog.in_edges(e_id)
    next_act = 0
    for (source,target) in out_edges:
        if ocel.get_value(target,"event_activity") == act:
            next_act = 1
            for (source_, target_) in out_edges:
                if ocel.get_value(target_, "event_timestamp")< ocel.get_value(source_, "event_timestamp"):
                    next_act = 0
    return next_act

NEXT_TIMESTAMP = "next_timestamp"
def next_timestamp(node,ocel,params):
    e_id = node.event_id
    out_edges = ocel.graph.eog.in_edges(e_id)
    if len(out_edges) == 0:
        return 0
    return min([ocel.get_value(target,"event_timestamp") for (source,target) in out_edges]).to_pydatetime().timestamp()

ocpa.algo.predictive_monitoring.factory.VERSIONS[ocpa.algo.predictive_monitoring.factory.EVENT_BASED][NEXT_ACTIVITY] = next_activity
ocpa.algo.predictive_monitoring.factory.VERSIONS[ocpa.algo.predictive_monitoring.factory.EVENT_BASED][NEXT_TIMESTAMP] = next_timestamp


def filter_process_executions(ocel, cases):
    '''
    Filters process executions from an ocel

    :param ocel: Object-centric event log
    :type ocel: :class:`OCEL <ocpa.objects.log.ocel.OCEL>`

    :param cases: list of cases to be included (index of the case property of the OCEL)
    :type threshold: list(int)

    :return: Object-centric event log
    :rtype: :class:`OCEL <ocpa.objects.log.ocel.OCEL>`

    '''

    events = [e for case in cases for e in case]
    new_event_df = ocel.log.log.loc[ocel.log.log["event_id"].isin(events)].copy()
    new_log = log_util.copy_log_from_df(new_event_df, ocel.parameters)
    return new_log

def cat_target_to_vector(feature_storage, targets, new_name):
    #targets needs to be ordered
    for g in feature_storage.feature_graphs:
        for node in g.nodes:
            vec = [node.attributes[t] for t in targets]
            vec = [1 if e == max(vec) else 0 for e in vec]
            node.attributes[new_name] = vec
            for t in targets:
                del node.attributes[t]
    return feature_storage


def GNN_prediction(layer_size, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=64, lr=0.01, n_output = 1):
    # return 0,0,0,0

    train_loader = GraphDataLoader(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        add_self_loop=True,
        make_bidirected=False,
        on_gpu=False
    )
    val_loader = GraphDataLoader(
        x_val,
        y_val,
        batch_size=batch_size,
        shuffle=True,
        add_self_loop=True,
        make_bidirected=False,
        on_gpu=False
    )
    test_loader = GraphDataLoader(
        x_test,
        y_test,
        batch_size=128,
        shuffle=False,
        add_self_loop=True,
        make_bidirected=False,
        on_gpu=False
    )

    # define GCN model
    tf.keras.backend.clear_session()
    model = None
    #regression
    if n_output == 1:
        model = GCN(layer_size, layer_size)
        optimizer = tf.keras.optimizers.Adam(lr=lr)
        loss_function = tf.keras.losses.MeanAbsoluteError()
    else:
        model = ClassificationGCN(layer_size,layer_size,n_output)
        optimizer = tf.keras.optimizers.Adam(lr=lr)
        loss_function = tf.keras.losses.MeanSquaredError()#CategoricalCrossentropy(from_logits=True)
    # run tensorflow training loop
    epochs = 20#3#30
    iter_idx = np.arange(0, train_loader.__len__())
    loss_history = []
    val_loss_history = []
    step_losses = []
    for e in range(epochs):
        print('Running epoch:', e)
        np.random.shuffle(iter_idx)
        current_loss = step = 0
        for batch_id in tqdm(iter_idx):
            step += 1
            dgl_batch, label_batch = train_loader.__getitem__(batch_id)
            #if n_output != 1:
            #    label_batch = tf.reshape(label_batch, (int(label_batch.shape[0] * label_batch.shape[1]),))
            with tf.GradientTape() as tape:
                pred = model(dgl_batch, dgl_batch.ndata['features'])
                #print(pred.shape)
                #if n_output != 1:
                #    pred = tf.reshape(pred, (int(pred.shape[0] * pred.shape[1]),))
                loss = loss_function(label_batch, pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            step_losses.append(loss.numpy())
            current_loss += loss.numpy()
            # if (step % 100 == 0): print('Loss: %s'%((current_loss / step)))
            loss_history.append(current_loss / step)
        val_predictions, val_labels = evaluate_gnn(val_loader, model)
        val_loss = 0
        if n_output== 1:
            val_loss = tf.keras.metrics.mean_absolute_error(np.squeeze(val_labels), np.squeeze(val_predictions)).numpy()
            print('    Validation MAE GNN:', val_loss)
            if len(val_loss_history) < 1:
                model.save_weights('gnn_checkpoint.tf')
                print('    GNN checkpoint saved.')
            else:
                if val_loss < np.min(val_loss_history):
                    model.save_weights('gnn_checkpoint.tf')
                    print('    GNN checkpoint saved.')
        else:
            m = tf.keras.metrics.CategoricalAccuracy()
            m.update_state(val_labels, val_predictions)
            m.result().numpy()
            val_loss = m.result().numpy()#tf.keras.metrics.categorical_accuracy(val_labels, val_predictions).numpy()

            print('    Validation Accuracy GNN:', val_loss)
            if len(val_loss_history) < 1:
                model.save_weights('gnn_checkpoint.tf')
                print('    GNN checkpoint saved.')
            else:
                if val_loss > np.min(val_loss_history):
                    model.save_weights('gnn_checkpoint.tf')
                    print('    GNN checkpoint saved.')
        val_loss_history.append(val_loss)

    # visualize training progress
    pd.DataFrame({'loss': loss_history, 'step_losses': step_losses}).plot(subplots=True, layout=(1, 2), sharey=True)

    # restore weights from best epoch
    cp_status = model.load_weights('gnn_checkpoint.tf')
    cp_status.assert_consumed()

    # generate predictions and calculate MAE for train, val & test sets
    train_predictions, train_labels = evaluate_gnn(train_loader, model)
    val_predictions, val_labels = evaluate_gnn(val_loader, model)
    test_predictions, test_labels = evaluate_gnn(test_loader, model)

    baseline = 0
    train_score = 0
    test_score = 0
    val_score = 0

    #regression
    if n_output==1:
        mean_prediction = np.mean(np.array(y_train))
        print('MAE baseline: ')
        baseline = mean_absolute_error(test_labels, np.repeat(mean_prediction, len(test_labels)))
        print(mean_absolute_error(test_labels, np.repeat(mean_prediction, len(test_labels))))
        print('MAE GNN: ')
        test_score = mean_absolute_error(test_predictions, test_labels)
        print(test_score)
        train_score = mean_absolute_error(train_predictions, train_labels)
        val_score= mean_absolute_error(val_predictions, val_labels)
        print(test_predictions)
        print(test_labels)
    #classification
    else:
        max_elem_prediction = [0 for i in range(0,n_output)]
        agg_elems = sum(train_labels)
        majority = list(agg_elems).index(max(list(agg_elems)))
        max_elem_prediction = [1 if i == majority else 0 for i in range(0,len(agg_elems))]
        #print(max_elem_prediction)
        #print(max_elem_prediction)
        print("Accuracy baseline: ")
        #baseline = accuracy_score(test_labels, np.repeat(max_elem_prediction, len(test_labels)).reshape(len(test_labels),len(max_elem_prediction)))
        baseline_predictions = [max_elem_prediction for i in range(0,len(test_labels))]
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(test_labels, baseline_predictions)
        baseline = m.result().numpy()
        print(baseline)
        print('Accuracy GNN: ')
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(test_labels, test_predictions)
        test_score = m.result().numpy()
        print(test_score)
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(train_labels, train_predictions)
        train_score = m.result().numpy()
        #train_predictions = [[1 if x > 0.5 else 0 for x in train_predictions_l] for train_predictions_l in train_predictions]
        #train_score = accuracy_score(train_predictions, train_labels)

        #val_predictions = [[1 if x > 0.5 else 0 for x in val_predictions_l] for val_predictions_l in
        #                     val_predictions]
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(val_labels, val_predictions)
        #m.result().numpy()
        val_score = m.result().numpy()
        #val_score = accuracy_score(val_predictions, val_labels)
        #print(test_predictions)
        #print(test_labels)

    return baseline,train_score,val_score,test_score

# filename = "BPI2017-Final.csv"
# object_types = ["application", "offer"]
# parameters = {"obj_names":object_types,
#               "val_names":[],
#               "act_name":"event_activity",
#               "time_name":"event_timestamp",
#               "sep":","}
# ocel = csv_import_factory.apply(file_path= filename,parameters = parameters)
#ocel = filter_process_executions(ocel, ocel.process_executions[0:200])


filename = "p2p-normal.jsonocel"
ocel = ocel_import_factory.apply(filename)

#filename = "running-example.jsonocel"
#parameters = {"execution_extraction": "leading_type",
#              "leading_type": "items"}
#ocel = ocel_import_factory.apply(filename)#, parameters = parameters)
#ocel = filter_process_executions(ocel, ocel.process_executions[0:200])


print("Number of process executions: "+str(len(ocel.process_executions)))
activities = list(set(ocel.log.log["event_activity"].tolist()))
print(str(len(activities))+" actvities")


for target in [[(NEXT_ACTIVITY,(act,)) for act in activities]]:
    include_last = False
    F = target+[(predictive_monitoring.EVENT_SYNCHRONIZATION_TIME, ())]
    feature_storage = predictive_monitoring.apply(ocel, F, [])
    # replace synchronization time with 0 placeholder for empty feature
    for g in feature_storage.feature_graphs:
        for n in g.nodes:
            n.attributes[('event_synchronization_time', ())] = 1
    feature_storage.extract_normalized_train_test_split(0.3, state=3)
    for g in feature_storage.feature_graphs:
        for n in g.nodes:
            n.attributes[('event_synchronization_time', ())] = 1
    accuracy_dict = {}

    #replace categorical features with vector
    new_target_name = (target[0][0],())
    feature_storage = cat_target_to_vector(feature_storage,target, new_target_name)
    for k in [4,5]:
        if True:
            print("___________________________")
            print("Prediction with Graph Structure and GNN")
            print("___________________________")

            layer_size = len(F)-len(target)

            # generate training & test datasets
            train_idx, val_idx = train_test_split(feature_storage.training_indices, test_size = 0.2)
            x_train, y_train = generate_graph_dataset(feature_storage.feature_graphs, train_idx, ocel, k = k, target = new_target_name, include_last = include_last)
            x_val, y_val = generate_graph_dataset(feature_storage.feature_graphs, val_idx, ocel, k = k,target = new_target_name, include_last = include_last)
            x_test, y_test = generate_graph_dataset(feature_storage.feature_graphs, feature_storage.test_indices, ocel, k = k,target = new_target_name, include_last = include_last)

            baseline_MAE, train_MAE, val_MAE, test_MAE = GNN_prediction(layer_size,x_train, y_train,x_val, y_val,x_test, y_test, batch_size=4, lr = 0.01, n_output=len(activities))
            # record performance of GNN
            accuracy_dict[new_target_name[0]+'graph_gnn_k_' + str(k)] = {
                'baseline_ACC': baseline_MAE,
                'train_ACC': train_MAE,
                'val_ACC': val_MAE,
                'test_ACC': test_MAE
            }
            print(pd.DataFrame(accuracy_dict))

        if True:
            print("___________________________")
            print("Prediction with Sequential Structure and GNN")
            print("___________________________")

            layer_size = len(F) - len(target)
            train_idx, val_idx = train_test_split(feature_storage.training_indices, test_size=0.2)
            x_train, y_train = generate_sequential_graph_dataset(feature_storage.feature_graphs, train_idx, ocel, k=k,
                                                                 target=new_target_name, include_last=include_last)
            x_val, y_val = generate_sequential_graph_dataset(feature_storage.feature_graphs, val_idx, ocel, k=k,
                                                             target=new_target_name, include_last=include_last)
            x_test, y_test = generate_sequential_graph_dataset(feature_storage.feature_graphs,
                                                               feature_storage.test_indices, ocel, k=k, target=new_target_name,
                                                               include_last=include_last)
            baseline_MAE, train_MAE, val_MAE, test_MAE = GNN_prediction(layer_size, x_train, y_train, x_val, y_val,
                                                                        x_test, y_test, batch_size=4, lr=0.01,
                                                                        n_output=len(activities))
            # record performance of GNN
            accuracy_dict[new_target_name[0] + '_flat_gnn_k_' + str(k)] = {
                'baseline_ACC': baseline_MAE,
                'train_ACC': train_MAE,
                'val_ACC': val_MAE,
                'test_ACC': test_MAE
            }
            print(pd.DataFrame(accuracy_dict))

ökahdlif
for target in [(NEXT_TIMESTAMP,()),(predictive_monitoring.EVENT_REMAINING_TIME,())]:
    include_last = True
    if target == (NEXT_TIMESTAMP,()):
        include_last = False

    # F = [(predictive_monitoring.EVENT_REMAINING_TIME,()),
    #      (predictive_monitoring.EVENT_PREVIOUS_TYPE_COUNT,("offer",)),
    #      (predictive_monitoring.EVENT_ELAPSED_TIME,())] + [(predictive_monitoring.EVENT_AGG_PREVIOUS_CHAR_VALUES,("event_RequestedAmount",max))] \
    #     + [(predictive_monitoring.EVENT_PRECEDING_ACTIVITES,(act,)) for act in activities]

    F = [target,
         (predictive_monitoring.EVENT_SYNCHRONIZATION_TIME, ())]#+ [(predictive_monitoring.EVENT_PRECEDING_ACTIVITES,(act,)) for act in activities] #,
        #+ [(predictive_monitoring.EVENT_PRECEDING_ACTIVITES,(act,)) for act in activities]
        # (predictive_monitoring.EVENT_FLOW_TIME,())]  \
        #+ [(predictive_monitoring.EVENT_PRECEDING_ACTIVITES,(act,)) for act in activities]

    # F = [(predictive_monitoring.EVENT_REMAINING_TIME,()),
    #     # (predictive_monitoring.EVENT_PREVIOUS_TYPE_COUNT,("GDSRCPT",)),
    #      (predictive_monitoring.EVENT_ELAPSED_TIME,())]  \
    #     + [(predictive_monitoring.EVENT_PRECEDING_ACTIVITES,(act,)) for act in activities]


    feature_storage = predictive_monitoring.apply(ocel, F, [])
    #replace synchronization time with 0 placeholder for empty feature
    for g in feature_storage.feature_graphs:
        for n in g.nodes:
            n.attributes[('event_synchronization_time',())] = 1
    feature_storage.extract_normalized_train_test_split(0.3,state = 3)
    for g in feature_storage.feature_graphs:
        for n in g.nodes:
            n.attributes[('event_synchronization_time',())] = 1
    accuracy_dict = {}



    g_set_list_t =[]
    g_set_list_te = []
    seq_set_list = []
    seq_set_list_v = []
    seq_set_list_t = []



    for k in [4]:
        if True:
            print("___________________________")
            print("Prediction with Graph Structure and GNN")
            print("___________________________")

            layer_size = len(F)-1

            # generate training & test datasets
            train_idx, val_idx = train_test_split(feature_storage.training_indices, test_size = 0.2)
            x_train, y_train = generate_graph_dataset(feature_storage.feature_graphs, train_idx, ocel, k = k, target = target, include_last = include_last)
            # dgl.save_graphs('train_graph_dataset', x_train, labels = {'remaining_time': tf.constant(y_train)})
            # x_train, y_train = dgl.load_graphs('train_graph_dataset')
            # y_train = y_train['remaining_time']
            x_val, y_val = generate_graph_dataset(feature_storage.feature_graphs, val_idx, ocel, k = k,target = target, include_last = include_last)
            # dgl.save_graphs('val_graph_dataset', x_val, labels = {'remaining_time': tf.constant(y_val)})
            # x_val, y_val = dgl.load_graphs('val_graph_dataset')
            # y_val = y_val['remaining_time']
            x_test, y_test = generate_graph_dataset(feature_storage.feature_graphs, feature_storage.test_indices, ocel, k = k,target = target, include_last = include_last)
            # dgl.save_graphs('test_graph_dataset', x_test, labels = {'remaining_time': tf.constant(y_test)})
            # x_test, y_test = dgl.load_graphs('test_graph_dataset')
            # y_test = y_test['remaining_time']
            # explore data instances
            ####for idx in [3]:
                ####visualize_graph(x_train[idx],str(idx)+"graph", labels='node_id')
                #visualize_graph(x_train[idx],str(idx)+"graph", labels='event_indices')
            # visualize_graph(x_train[idx], labels = 'remaining_time')
            # show_remaining_times(x_train[idx])
            ####visualize_instance(x_train[idx], "graph", y_train[idx])
            # get_ordered_event_list(x_train[idx])['events']
            # get_ordered_event_list(x_train[idx])['features']

            baseline_MAE, train_MAE, val_MAE, test_MAE = GNN_prediction(layer_size,x_train, y_train,x_val, y_val,x_test, y_test, batch_size=4, lr = 0.005)
            # record performance of GNN
            accuracy_dict[target[0]+'graph_gnn_k_' + str(k)] = {
                'baseline_MAE': baseline_MAE,
                'train_MAE': train_MAE,
                'val_MAE': val_MAE,
                'test_MAE': test_MAE
            }
            print(pd.DataFrame(accuracy_dict))




        if True:
            print("___________________________")
            print("Prediction with Sequential Structure and GNN")
            print("___________________________")

            layer_size = len(F)-1



            # generate training & test datasets
            train_idx, val_idx = train_test_split(feature_storage.training_indices, test_size = 0.2)
            x_train, y_train = generate_sequential_graph_dataset(feature_storage.feature_graphs, train_idx, ocel, k = k, target = target, include_last = include_last)
            # dgl.save_graphs('train_graph_dataset', x_train, labels = {'remaining_time': tf.constant(y_train)})
            # x_train, y_train = dgl.load_graphs('train_graph_dataset')
            # y_train = y_train['remaining_time']
            x_val, y_val = generate_sequential_graph_dataset(feature_storage.feature_graphs, val_idx, ocel, k = k, target = target, include_last = include_last)

            # dgl.save_graphs('val_graph_dataset', x_val, labels = {'remaining_time': tf.constant(y_val)})
            # x_val, y_val = dgl.load_graphs('val_graph_dataset')
            # y_val = y_val['remaining_time']
            x_test, y_test = generate_sequential_graph_dataset(feature_storage.feature_graphs, feature_storage.test_indices, ocel, k = k, target = target, include_last = include_last)


            # dgl.save_graphs('test_graph_dataset', x_test, labels = {'remaining_time': tf.constant(y_test)})
            # x_test, y_test = dgl.load_graphs('test_graph_dataset')
            # y_test = y_test['remaining_time']
            # explore data instances
            ###for idx in [3]:
            ###    visualize_graph(x_train[idx],str(idx)+"flat", labels='node_id')
                #visualize_graph(x_train[idx],str(idx)+"flat", labels='event_indices')
            # visualize_graph(x_train[idx], labels = 'remaining_time')
            # show_remaining_times(x_train[idx])
            ###visualize_instance(x_train[idx],"flat", y_train[idx])
            # get_ordered_event_list(x_train[idx])['events']
            # get_ordered_event_list(x_train[idx])['features']

            baseline_MAE, train_MAE, val_MAE, test_MAE = GNN_prediction(layer_size, x_train, y_train, x_val, y_val, x_test,
                                                                        y_test, batch_size=4, lr = 0.005)
            # record performance of GNN
            accuracy_dict[target[0]+'flat_gnn_k_' + str(k)] = {
                'baseline_MAE': baseline_MAE,
                'train_MAE': train_MAE,
                'val_MAE': val_MAE,
                'test_MAE': test_MAE
            }
            print(pd.DataFrame(accuracy_dict))

        if True:
            print("___________________________")
            print("Prediction with Graph Embedding")
            print("___________________________")
            train_nx_feature_graphs = []
            test_nx_feature_graphs = []
            train_target = []
            test_target = []
            print("Constructing Subgraphs ")
            for i in tqdm(feature_storage.training_indices):
                g = feature_storage.feature_graphs[i]
                converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                              target=target,
                                                                              from_start=False, include_last = include_last)

                train_nx_feature_graphs += converted_subgraphs
                train_target += extracted_targets

            for i in tqdm(feature_storage.test_indices):
                g = feature_storage.feature_graphs[i]
                converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                              target=target,
                                                                              from_start=False, include_last = include_last)
                test_nx_feature_graphs += converted_subgraphs
                test_target += extracted_targets

            # IGE has problems with sparseness
            for embedding_technique in ['FEATHER-G', 'Graph2Vec', 'NetLSD', 'WaveletCharacteristic',
                                        # 'IGE',
                                        'LDP', 'GL2Vec', 'SF', 'FGSD']:  # , 'TAIWAN']:
                try:
                    X_train, X_test = embed(train_nx_feature_graphs, test_nx_feature_graphs, embedding_technique,size = 10*k)
                    print(X_train.shape)
                    print(X_test.shape)
                    model = LinearRegression()
                    model.fit(X_train, train_target)
                    res = model.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    accuracy_dict[target[0]+'embed_reg_'+ embedding_technique+'_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }
                    regr = MLPRegressor(random_state=3, max_iter=500,hidden_layer_sizes=(10,10,)).fit(X_train, train_target)
                    res = regr.predict(X_test)
                    print(mean_absolute_error(test_target, res))
                    accuracy_dict[target[0]+'embed_nn_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': mean_absolute_error(test_target, res)
                    }
                except ValueError:
                    accuracy_dict[target[0]+'embed_reg_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': "NA"
                    }
                    accuracy_dict[target[0] + 'embed_nn_' + embedding_technique + '_k_' + str(k)] = {
                        'baseline_MAE': 0,
                        'train_MAE': 0,
                        'val_MAE': 0,
                        'test_MAE': "NA"
                    }
            print(pd.DataFrame(accuracy_dict))



    pd.DataFrame(accuracy_dict).to_csv("results_"+target[0]+".csv")

