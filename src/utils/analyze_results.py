import pandas as pd

""""
File for single seed experiments results
"""
df = pd.DataFrame(columns=['run',
                           'model',
                           'dataset_name',
                           'emb_dim',
                           'lags',
                           'prediction_window',
                           'val_MSE',
                           'val_MAE',
                           'test_MSE',
                           'test_MAE'])

with open('../../logs/logs_PV_final.txt', 'r') as f:
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
        val_loss = float(line_list[-7])
        val_acc = float(line_list[-5])
        test_loss = float(line_list[-3])
        test_acc = float(line_list[-1])

        cont += 1
        new_row = [run, model, dataset_name, emb_dim, lags, prediction_window, val_loss, val_acc, test_loss, test_acc]
        new_row_df = pd.DataFrame([new_row], columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)


print('ok')

# Utils command
# top_10_rows = df.nlargest(10, 'B')

# a = df.loc[df['use_second_test_emb'] == 'False']
# a.nlargest(10, 'test_acc').to_excel("temp.xlsx", index=False)
# a.to_excel("temp.xlsx", index=False)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)