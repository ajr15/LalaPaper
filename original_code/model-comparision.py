import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from rdkit import Chem
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split

from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec



def compare_models(X, y, estimators:list, plotcolor:str, filename:str):
    fig, axs = plt.subplots(2, 3, figsize=(12, 9))
    axs = np.ravel(axs)

    for ax, (name, est) in zip(axs, estimators):
        start_time = time.time()
        score = cross_validate(est, X, y,
                            scoring=['r2', 'neg_mean_absolute_error'],
                            n_jobs=-1, verbose=0)
        elapsed_time = time.time() - start_time

        y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0)
        plot_regression_results(
            ax, y, y_pred, name,
            (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$MAE={:.2f} \pm {:.2f}$')
            .format(np.mean(score['test_r2']),
                    np.std(score['test_r2']),
                    -np.mean(score['test_neg_mean_absolute_error']),
                    np.std(score['test_neg_mean_absolute_error'])),
            elapsed_time, plotcolor)

    plt.suptitle('Comparing Predictors')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(filename)
    # plt.show()

def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time, plotcolor):
    """Scatter plot of the predicted vs true targets."""
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
    ax.legend([extra], [scores], loc='upper left', fontsize=12, 
              borderpad=0.1, borderaxespad=0.1, handlelength=0)
    title = title + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)
    ax.set_title(title)

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
    # mdf = mdf.reset_index()

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

    # models of interest: Lasso/ElasticNet (few important features); RidgeRegression/SVR(kernel='linear') or SVR(kernel='rbf')/EnsembleRegressors (if previous does not work)
    # choosing the estimators to compare
    estimators = [
        ('Lasso', LassoCV()),
        ('ElasticNet', ElasticNetCV()),
        ('Ridge', RidgeCV()),
        ('Support Vector - linear kernel', SVR(kernel='linear', max_iter=10000)),
        ('Support Vector - rbf kernel', SVR()),
        ('Random Forest', RandomForestRegressor(random_state=42))
    ]


    # assign X and y
    # properties to predict: Gap, HOMO, LUMO, IP, EA, relative energy
    properties = ['HOMO_eV', 'LUMO_eV', 'GAP_eV', 'aEA_eV', 'aIP_eV', 'Erel_eV']

    # here we do mol2vec and we predict properties
    X = pd.DataFrame(np.array([x.vec for x in mdf_train['mol2vec']]))
    print(X) 

    for property in properties:
        y = properties_train[property].values
        print(f'{property}: {y}')
        filename = "figures/comparing-models-QSPR/comparing-models-mol2vec-"+property+".png"
        compare_models(X, y, estimators, 'dodgerblue', filename)

    # just all structural features
    X = df_structural[['longest_L', 'ratio_L', 'n_LAL', 'longest_A', 'n_branches', 'n_rings']].copy()
    X = features_train.drop(['annulation'], axis=1)
    print(X) 

    for property in properties:
        y = properties_train[property].values
        print(f'{property}: {y}')
        filename = "figures/comparing-models-QSPR/comparing-models-structfeat-"+property+".png"
        compare_models(X, y, estimators, 'royalblue', filename)

    # mol2vec and structural features
    m2v = pd.DataFrame(np.array([x.vec for x in mdf_train['mol2vec']]), index=mdf_train.index)
    features = features_train.drop(['annulation'], axis=1)
    X = pd.concat((m2v, features), axis=1)
    print(X) 

    for property in properties:
        y = properties_train[property].values
        print(f'{property}: {y}')
        filename = "figures/comparing-models-QSPR/comparing-models-mol2vec-structfeat-"+property+".png"
        compare_models(X, y, estimators, 'teal', filename)
