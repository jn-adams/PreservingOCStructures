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

params = {"sap": {"batch_size":4,"lr":0.001,"epochs":15}}


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


def GNN_prediction(layer_size, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=64, lr=0.01):
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
    model = GCN(layer_size, layer_size)
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    loss_function = tf.keras.losses.MeanAbsoluteError()

    # run tensorflow training loop
    epochs = 12#3#30
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
            with tf.GradientTape() as tape:
                pred = model(dgl_batch, dgl_batch.ndata['features'])
                loss = loss_function(label_batch, pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            step_losses.append(loss.numpy())
            current_loss += loss.numpy()
            # if (step % 100 == 0): print('Loss: %s'%((current_loss / step)))
            loss_history.append(current_loss / step)
        val_predictions, val_labels = evaluate_gnn(val_loader, model)
        val_loss = tf.keras.metrics.mean_absolute_error(np.squeeze(val_labels), np.squeeze(val_predictions)).numpy()
        print('    Validation MAE GNN:', val_loss)
        if len(val_loss_history) < 1:
            model.save_weights('gnn_checkpoint.tf')
            print('    GNN checkpoint saved.')
        else:
            if val_loss < np.min(val_loss_history):
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
    mean_prediction = np.mean(np.array(y_train))
    print('MAE baseline: ')
    baseline = mean_absolute_error(test_labels, np.repeat(mean_prediction, len(test_labels)))
    print(mean_absolute_error(test_labels, np.repeat(mean_prediction, len(test_labels))))
    print('MAE GNN: ')
    print(mean_absolute_error(test_predictions, test_labels))

    print(test_predictions)
    print(test_labels)

    return baseline,mean_absolute_error(train_predictions, train_labels),mean_absolute_error(val_predictions, val_labels),mean_absolute_error(test_predictions, test_labels)

# filename = "BPI2017-Final.csv"
# object_types = ["application", "offer"]
# parameters = {"obj_names":object_types,
#               "val_names":[],
#               "act_name":"event_activity",
#               "time_name":"event_timestamp",
#               "sep":","}
# ocel = csv_import_factory.apply(file_path= filename,parameters = parameters)
# ocel = filter_process_executions(ocel, ocel.process_executions[0:1000])


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
# F = [(predictive_monitoring.EVENT_REMAINING_TIME,()),
#      (predictive_monitoring.EVENT_PREVIOUS_TYPE_COUNT,("offer",)),
#      (predictive_monitoring.EVENT_ELAPSED_TIME,())] + [(predictive_monitoring.EVENT_AGG_PREVIOUS_CHAR_VALUES,("event_RequestedAmount",max))] \
#     + [(predictive_monitoring.EVENT_PRECEDING_ACTIVITES,(act,)) for act in activities]

F = [(predictive_monitoring.EVENT_REMAINING_TIME,()),
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








for k in [4,5]:
    if True:
        print("___________________________")
        print("Prediction with Graph Structure and GNN")
        print("___________________________")

        layer_size = len(F)-1

        # generate training & test datasets
        train_idx, val_idx = train_test_split(feature_storage.training_indices, test_size = 0.2)
        x_train, y_train = generate_graph_dataset(feature_storage.feature_graphs, train_idx, ocel, k = k)
        # dgl.save_graphs('train_graph_dataset', x_train, labels = {'remaining_time': tf.constant(y_train)})
        # x_train, y_train = dgl.load_graphs('train_graph_dataset')
        # y_train = y_train['remaining_time']
        x_val, y_val = generate_graph_dataset(feature_storage.feature_graphs, val_idx, ocel, k = k)
        # dgl.save_graphs('val_graph_dataset', x_val, labels = {'remaining_time': tf.constant(y_val)})
        # x_val, y_val = dgl.load_graphs('val_graph_dataset')
        # y_val = y_val['remaining_time']
        x_test, y_test = generate_graph_dataset(feature_storage.feature_graphs, feature_storage.test_indices, ocel, k = k)
        # dgl.save_graphs('test_graph_dataset', x_test, labels = {'remaining_time': tf.constant(y_test)})
        # x_test, y_test = dgl.load_graphs('test_graph_dataset')
        # y_test = y_test['remaining_time']
        # explore data instances
        for idx in [3]:
            visualize_graph(x_train[idx],str(idx)+"graph", labels='node_id')
            #visualize_graph(x_train[idx],str(idx)+"graph", labels='event_indices')
        # visualize_graph(x_train[idx], labels = 'remaining_time')
        # show_remaining_times(x_train[idx])
        visualize_instance(x_train[idx], "graph", y_train[idx])
        # get_ordered_event_list(x_train[idx])['events']
        # get_ordered_event_list(x_train[idx])['features']

        baseline_MAE, train_MAE, val_MAE, test_MAE = GNN_prediction(layer_size,x_train, y_train,x_val, y_val,x_test, y_test, batch_size=4, lr = 0.005)
        # record performance of GNN
        accuracy_dict['graph_gnn_k_' + str(k)] = {
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
        x_train, y_train = generate_sequential_graph_dataset(feature_storage.feature_graphs, train_idx, ocel, k = k)
        # dgl.save_graphs('train_graph_dataset', x_train, labels = {'remaining_time': tf.constant(y_train)})
        # x_train, y_train = dgl.load_graphs('train_graph_dataset')
        # y_train = y_train['remaining_time']
        x_val, y_val = generate_sequential_graph_dataset(feature_storage.feature_graphs, val_idx, ocel, k = k)
        # dgl.save_graphs('val_graph_dataset', x_val, labels = {'remaining_time': tf.constant(y_val)})
        # x_val, y_val = dgl.load_graphs('val_graph_dataset')
        # y_val = y_val['remaining_time']
        x_test, y_test = generate_sequential_graph_dataset(feature_storage.feature_graphs, feature_storage.test_indices, ocel, k = k)
        # dgl.save_graphs('test_graph_dataset', x_test, labels = {'remaining_time': tf.constant(y_test)})
        # x_test, y_test = dgl.load_graphs('test_graph_dataset')
        # y_test = y_test['remaining_time']
        # explore data instances
        for idx in [3]:
            visualize_graph(x_train[idx],str(idx)+"flat", labels='node_id')
            #visualize_graph(x_train[idx],str(idx)+"flat", labels='event_indices')
        # visualize_graph(x_train[idx], labels = 'remaining_time')
        # show_remaining_times(x_train[idx])
        visualize_instance(x_train[idx],"flat", y_train[idx])
        # get_ordered_event_list(x_train[idx])['events']
        # get_ordered_event_list(x_train[idx])['features']

        baseline_MAE, train_MAE, val_MAE, test_MAE = GNN_prediction(layer_size, x_train, y_train, x_val, y_val, x_test,
                                                                    y_test, batch_size=4, lr = 0.005)
        # record performance of GNN
        accuracy_dict['flat_gnn_k_' + str(k)] = {
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
        for i in feature_storage.training_indices:
            g = feature_storage.feature_graphs[i]
            converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                          target=("event_remaining_time", ()),
                                                                          from_start=False)
            train_nx_feature_graphs += converted_subgraphs
            train_target += extracted_targets

        for i in feature_storage.training_indices:
            g = feature_storage.feature_graphs[i]
            converted_subgraphs, extracted_targets = convert_to_nx_graphs(g, ocel, k,
                                                                          target=("event_remaining_time", ()),
                                                                          from_start=False)
            test_nx_feature_graphs += converted_subgraphs
            test_target += extracted_targets

        # IGE has problems with sparseness
        for embedding_technique in ['FEATHER-G', 'Graph2Vec', 'NetLSD', 'WaveletCharacteristic',
                                    # 'IGE',
                                    'LDP', 'GL2Vec', 'SF', 'FGSD']:  # , 'TAIWAN']:
            X_train, X_test = embed(train_nx_feature_graphs, test_nx_feature_graphs, embedding_technique)
            print(X_train.shape)
            print(X_test.shape)
            model = LinearRegression()
            model.fit(X_train, train_target)
            res = model.predict(X_test)
            print(mean_absolute_error(test_target, res))
            accuracy_dict['embed_reg_'+ embedding_technique+'_k_' + str(k)] = {
                'baseline_MAE': 0,
                'train_MAE': 0,
                'val_MAE': 0,
                'test_MAE': mean_absolute_error(test_target, res)
            }

pd.DataFrame(accuracy_dict).to_csv("results.csv")

