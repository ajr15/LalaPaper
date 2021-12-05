import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from rdkit import Chem

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec




def plot_evaluation(y_pred, y_true, plotcolor, Xtype, property):

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    scores = (r'$R^2={:.2f}$' + '\n' + r'$MAE={:.2f}$').format(r2, mae)

    fig, ax = plt.subplots(figsize=[4,4])
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2, color=plotcolor)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # ax.set_xlim([y_true.min(), y_true.max()])
    # ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Computed')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("figures/RF-test-"+Xtype+"-"+property+".png")
    plt.clf()

def optimizeRF(X_train, X_test, y_train, y_test, features, property:str, Xtype:str):

    ntrees_list = np.arange(1,30)

    train_errors = []
    test_errors = []

    for ntrees in ntrees_list:

        rfr = RandomForestRegressor(n_estimators=ntrees, random_state=42)
        rfr.fit(X_train, y_train)
        train_preds = rfr.predict(X_train)
        test_preds = rfr.predict(X_test)

        train_error = mean_squared_error(train_preds, y_train)
        train_errors.append(train_error)
        
        test_error = mean_squared_error(test_preds, y_test)
        test_errors.append(test_error)

    N_min_test_error = test_errors.index(min(test_errors)) + 1
    print(N_min_test_error)
    fig, ax = plt.subplots(figsize=[9,4])
    ax.plot(ntrees_list, train_errors, label='Train MSE')
    ax.vlines(N_min_test_error, 0, max(test_errors), label='Best N tree', edgecolor='black')
    ax.plot(ntrees_list, test_errors, label='Test MSE')
    ax.set_ylabel('MSE')
    ax.set_xlabel('N trees')
    ax.legend()
    plt.savefig("figures/optimizing-RF-"+Xtype+"-"+property+".png")
    plt.clf()

    return N_min_test_error
    


if __name__ == "__main__":

    # import data
    df_calculated = pd.read_csv('data/calculated_features.csv', index_col=0)
    df_structural = pd.read_csv('data/structural_features.csv', index_col=0)
    df_calculated.rename(columns={'Unnamed: 0': 'molecule'}, inplace=True)
    df_structural.rename(columns={'Unnamed: 0': 'molecule'}, inplace=True)

    mdf = pd.read_csv('data/goodstructures_smiles_noLin5.csv', sep=' ')
    mdf = mdf[~mdf.molecule.str.contains('c46h26')].reset_index(drop=True)
    mdf = mdf[~mdf.molecule.str.contains('c6h6')].reset_index(drop=True)
    mdf = mdf[~mdf.molecule.str.contains('c10h8')].reset_index(drop=True)
    mdf = mdf.set_index('molecule')
    mdf = mdf.reindex(index=df_calculated.index)

    print(mdf.head(5))
    print(df_structural.head(5))
    print(df_calculated.head(5))

    # trying mol2vec
    # 1. get mols from rdkit
    mdf['mol'] = mdf['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

    # 2. get mol2vec
    # Loading pre-trained model via word2vec
    model = word2vec.Word2Vec.load('model_300dim.pkl')
    print('Molecular sentence:', mol2alt_sentence(mdf['mol'][1], radius=1))
    print('\nMolSentence object:', MolSentence(mol2alt_sentence(mdf['mol'][1], radius=1)))
    print('\nmdfVec object:', DfVec(sentences2vec(mol2alt_sentence(mdf['mol'][1], radius=1), model, unseen='UNK')))

    # Constructing sentences
    mdf['sentence'] = mdf.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)

    # Extracting embeddings to a numpy.array
    # Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
    mdf['mol2vec'] = [DfVec(x) for x in sentences2vec(mdf['sentence'], model, unseen='UNK')]

    # split data into train and tes
    molecules = mdf.index
    print(molecules)
    train_molecules, test_molecules = train_test_split(molecules, test_size=0.2, random_state=42)

    features_train = df_structural.loc[train_molecules].copy()
    features_test = df_structural.loc[test_molecules].copy()
    print(features_test)
    properties_train = df_calculated.loc[train_molecules].copy()
    properties_test = df_calculated.loc[test_molecules].copy()
    print(properties_test)
    mdf_train = mdf.loc[train_molecules].copy()
    mdf_test = mdf.loc[test_molecules].copy()
    print(mdf_test)

    # model of interest chosen is random forest
    # optimizing RF N trees

    # assign X and y
    # properties to predict: Gap, HOMO, LUMO, IP, EA, relative energy
    properties = ['HOMO_eV', 'LUMO_eV', 'GAP_eV', 'aEA_eV', 'aIP_eV', 'Erel_eV']

    # here we do mol2vec and we predict properties
    X_train = pd.DataFrame(np.array([x.vec for x in mdf_train['mol2vec']]))
    X_test = pd.DataFrame(np.array([x.vec for x in mdf_test['mol2vec']]))
    print(X_train) 
    
    ntrees = [29,29,29,29,15,29]

    for property, n in zip(properties, ntrees):
        y_train = properties_train[property].values
        y_test = properties_test[property].values
        print(f'{property}: {y_train}')

        # nbranches = optimizeRF(X_train, X_test, y_train, y_test, X_train.columns, property, 'mol2vec')
        # rfr = RandomForestRegressor(n_estimators=nbranches, random_state=42)
        rfr = RandomForestRegressor(n_estimators=n, random_state=42)
        rfr.fit(X_train, y_train)
        y_pred = rfr.predict(X_test)

        plot_evaluation(y_pred, y_test, 'dodgerblue', 'mol2vec', property)

    # just all structural features
    # X_train = df_structural[['longest_L', 'ratio_L', 'n_LAL', 'longest_A', 'n_branches', 'n_rings']].copy()
    X_train = features_train.drop(['annulation'], axis=1)
    X_test = features_test.drop(['annulation'], axis=1)
    print(X_train) 

    df_feature_importance = pd.DataFrame(index=X_train.columns)
    features = X_train.columns

    ntrees = [24,29,29,28,26,20]

    for property, n in zip(properties, ntrees):
        y_train = properties_train[property].values
        y_test = properties_test[property].values
        print(f'{property}: {y_train}')

        # nbranches = optimizeRF(X_train, X_test, y_train, y_test, features, property, 'structfeat')
        # rfr = RandomForestRegressor(n_estimators=nbranches, random_state=42)
        rfr = RandomForestRegressor(n_estimators=n, random_state=42)
        rfr.fit(X_train, y_train)
        y_pred = rfr.predict(X_test)

        plot_evaluation(y_pred, y_test, 'royalblue', 'structfeat', property)

    #     feature_importances = rfr.feature_importances_
    #     df_feature_importance[property] = feature_importances
    #     feature_importances, features = zip(*sorted((zip(feature_importances, features))))

    #     fig, ax = plt.subplots(figsize=[8,4])
    #     ax.barh(features, feature_importances)
    #     ax.set_xlabel('Feature importances')
    #     plt.tight_layout()
    #     plt.savefig("figures/RF-structfeat-feature_importance-"+property+".png")
    #     plt.clf()
    
    # df_feature_importance.to_csv('data/rf_structfeat_feature_importance.csv')

    # mol2vec and structural features
    m2v = pd.DataFrame(np.array([x.vec for x in mdf_train['mol2vec']]), index=mdf_train.index)
    features = features_train.drop(['annulation'], axis=1)
    X_train = pd.concat((m2v, features), axis=1)
    print(X_train)
    m2v = pd.DataFrame(np.array([x.vec for x in mdf_test['mol2vec']]), index=mdf_test.index)
    features = features_test.drop(['annulation'], axis=1)
    X_test = pd.concat((m2v, features), axis=1) 

    df_feature_importance = pd.DataFrame(index=X_train.columns)
    features = X_train.columns

    ntrees = [29,27,28,28,29,29]

    for property, n in zip(properties, ntrees):
        y_train = properties_train[property].values
        y_test = properties_test[property].values
        print(f'{property}: {y_train}')

        # nbranches = optimizeRF(X_train, X_test, y_train, y_test, X_train.columns, property, 'mol2vec-structfeat')
        # rfr = RandomForestRegressor(n_estimators=nbranches, random_state=42)
        rfr = RandomForestRegressor(n_estimators=n, random_state=42)
        rfr.fit(X_train, y_train)
        y_pred = rfr.predict(X_test)

        plot_evaluation(y_pred, y_test, 'teal', 'mol2vec-structfeat', property)

        # feature_importances = rfr.feature_importances_
        # df_feature_importance[property] = feature_importances
        # feature_importances, features = zip(*sorted((zip(feature_importances, features))))

        # fig, ax = plt.subplots(figsize=[8,4])
        # ax.barh(features, feature_importances)
        # ax.set_xlabel('Feature importances')
        # plt.tight_layout()
        # plt.savefig("figures/RF-structfeat-feature_importance-"+property+".png")
        # plt.clf()
    
    # df_feature_importance.to_csv('data/rf_mol2vec_structfeat_feature_importance.csv')
