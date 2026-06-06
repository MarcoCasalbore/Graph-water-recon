import pandas as pd

""""
File for single seed experiments results
"""
df = pd.DataFrame(columns=['run',
                           'model',
                           'dataset_name',
                           'emb_dim',
                           'k',
                           'batch_size',
                           'lags',
                           'prediction_window',
                           'val_mse_mean',
                           'val_mse_std',
                           'val_rmse_mean',
                           'val_rmse_std',
                           'val_mae_mean',
                           'val_mae_std',
                           'val_mape_mean',
                           'val_mape_std',
                           'test_mse_mean',
                           'test_mse_std',
                           'test_rmse_mean',
                           'test_rmse_std',
                           'test_mae_mean',
                           'test_mae_std',
                           'test_mape_mean',
                           'test_mape_std'])

with open('../../logs/old logs/logs_METR-LA_mean_std_12_12 (Copy).txt', 'r') as f:
    train_list = list()
    test_list = list()
    cont = -1
    for line in f:
        line_list = line.strip().split()
        print(line_list)

        # Hyperparams
        run = int(line_list[1])
        model = line_list[3]
        dataset_name = line_list[5]
        emb_dim = int(line_list[7])
        k = int(line_list[9])
        batch_size = int(line_list[11])
        lags = int(line_list[13])
        prediction_window = int(line_list[15])

        # Metrics
        val_mse_mean = float(line_list[-31])
        val_mse_std = float(line_list[-29])
        val_mae_mean = float(line_list[-27])
        val_mae_std = float(line_list[-25])
        test_mse_mean = float(line_list[-23])
        test_mse_std = float(line_list[-21])
        test_mae_mean = float(line_list[-19])
        test_mae_std = float(line_list[-17])
        val_rmse_mean = float(line_list[-15])
        val_rmse_std = float(line_list[-13])
        val_mape_mean = float(line_list[-11])
        val_mape_std = float(line_list[-9])
        test_rmse_mean = float(line_list[-7])
        test_rmse_std = float(line_list[-5])
        test_mape_mean = float(line_list[-3])
        test_mape_std = float(line_list[-1])

        cont += 1
        new_row = [run,
                   model,
                   dataset_name,
                   emb_dim,
                   k,
                   batch_size,
                   lags,
                   prediction_window,
                   val_mse_mean,
                   val_mse_std,
                   val_rmse_mean,
                   val_rmse_std,
                   val_mae_mean,
                   val_mae_std,
                   val_mape_mean,
                   val_mape_std,
                   test_mse_mean,
                   test_mse_std,
                   test_rmse_mean,
                   test_rmse_std,
                   test_mae_mean,
                   test_mae_std,
                   test_mape_mean,
                   test_mape_std]
        new_row_df = pd.DataFrame([new_row], columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)

print('ok')

val_cols_target = ['model', 'emb_dim', 'k', 'batch_size', 'val_mse_mean', 'val_mse_std', 'val_rmse_mean',
                   'val_rmse_std', 'val_mae_mean', 'val_mae_std', 'val_mape_mean', 'val_mape_std']

test_cols_target = ['model', 'emb_dim', 'k', 'batch_size', 'test_mse_mean', 'test_mse_std', 'test_rmse_mean',
                   'test_rmse_std', 'test_mae_mean', 'test_mae_std', 'test_mape_mean', 'test_mape_std']

val_df = df[val_cols_target]
test_df = df[test_cols_target]
val_df = val_df.round(4)
test_df = test_df.round(4)

models_used = list(set(val_df['model']))
id_best_rows_dict =  {elemento: None for elemento in models_used}
for m in models_used:
    df_filtrato = df[df['model'] == m]
    indice_val_mse_min = df_filtrato['val_mse_mean'].idxmin()
    id_best_rows_dict[m] = indice_val_mse_min

val_best_idx_per_model = list(id_best_rows_dict.values())

print('val results!')
for index, row in val_df.iterrows():
    list_of_values = row.to_list()
    for id, elem in enumerate(list_of_values):
        if isinstance(elem, str):
            if '_' in elem:
                list_of_values[id] = elem.replace('_','-')
    if index not in val_best_idx_per_model:
        stringa_concatenata = ' & '.join([str(elem) for elem in list_of_values])
    else:
        stringa_concatenata = ' & '.join(['\\textbf{'+str(elem)+'}' for elem in list_of_values])
    print(stringa_concatenata + '\\\\')

print('test results!')
for index, row in test_df.iterrows():
    list_of_values = row.to_list()
    for id, elem in enumerate(list_of_values):
        if isinstance(elem, str):
            if '_' in elem:
                list_of_values[id] = elem.replace('_', '-')
    if index not in val_best_idx_per_model:
        stringa_concatenata = ' & '.join([str(elem) for elem in list_of_values])
    else:
        stringa_concatenata = ' & '.join(['\\textbf{' + str(elem) + '}' for elem in list_of_values])
    print(stringa_concatenata + '\\\\')

# Utils command
# top_10_rows = df.nlargest(10, 'B')

# a = df.loc[df['use_second_test_emb'] == 'False']
# a.nlargest(10, 'test_acc').to_excel("temp.xlsx", index=False)
# a.to_excel("temp.xlsx", index=False)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
