
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC                         # 只用這個是二分類
from sklearn.multiclass import OneVsRestClassifier  # 加這個才多分類

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


'''
# 計算指標
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
kappa = cohen_kappa_score(y_test, y_pred)
'''





def LDA(X_train_emb, y_train, 
        X_test_emb, y_test,
        ):

    # LDA 分類
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_emb, y_train)
    y_pred = lda.predict(X_test_emb)

    # 準確率
    acc = accuracy_score(y_test, y_pred)
    
    print("Fold accuracy:", acc)
    return acc

    

def SVM(X_train_emb, y_train,
        X_test_emb, y_test,
        kernels,
        acc_results):
    

    # kernels = ['linear', 'rbf', 'poly']   # 可比較的 kernel 種類
    
    # =====================================================
    # 🔹 逐個 kernel 訓練 SVM
    # =====================================================
    for kernel in kernels:
        clf = OneVsRestClassifier(SVC(kernel=kernel, C=0.6, gamma='scale'))
        clf.fit(X_train_emb, y_train)
        y_pred = clf.predict(X_test_emb)

        acc = accuracy_score(y_test, y_pred)
        acc_results[kernel].append(acc)
        print(f"SVM ({kernel}) accuracy: {acc:.4f}")


    return acc_results
    



"""

# func no use

def LDA(X, y):


    X, y = process_before_fold(X, y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    acc_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"===== Fold {fold+1} =====")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]


        '''MEAN = np.mean(X_train, axis=2, keepdims=True)
        STD = (np.std(X_train, axis=2, keepdims=True) + 1e-10)

        print(MEAN.shape)
        print(STD.shape)

        X_train = (X_train - MEAN) / STD
        X_test = (X_test - MEAN) / STD'''

        X_train_emb, X_test_emb = embedding(X_train, y_train, X_test, 
                                            embedding_method='Riemannian',
                                            n_components=5)

        # LDA 分類
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_emb, y_train)
        y_pred = lda.predict(X_test_emb)

        # 準確率
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)
        print("Fold accuracy:", acc)

    print()
    print("########################################################################")
    print("LDA result")
    print()
    print("平均準確率:", np.mean(acc_scores))
    print("########################################################################")
    print()

    '''
    


    # 計算指標
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)'''

def SVM(X, y):
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X, y = process_before_fold(X, y)

    kernels = ['linear', 'rbf', 'poly']   # 可比較的 kernel 種類
    acc_results = {k: [] for k in kernels}


    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"===== Fold {fold+1} =====")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]


        '''MEAN = np.mean(X_train, axis=2, keepdims=True)
        STD = (np.std(X_train, axis=2, keepdims=True) + 1e-10)

        print(MEAN.shape)
        print(STD.shape)

        X_train = (X_train - MEAN) / STD
        X_test = (X_test - MEAN) / STD'''

        
        # CSP 特徵提取
        '''csp = CSP(n_components=5, reg='ledoit_wolf', log=True, norm_trace=True)
        X_train_csp = csp.fit_transform(X_train, y_train)
        X_test_csp = csp.transform(X_test)'''


        X_train_emb, X_test_emb = embedding(X_train, y_train, X_test, 
                                            embedding_method='Riemannian',
                                            n_components=5)

        

        # =====================================================
        # 🔹 逐個 kernel 訓練 SVM
        # =====================================================
        for kernel in kernels:
            clf = OneVsRestClassifier(SVC(kernel=kernel, C=0.6, gamma='scale'))
            clf.fit(X_train_emb, y_train)
            y_pred = clf.predict(X_test_emb)

            acc = accuracy_score(y_test, y_pred)
            acc_results[kernel].append(acc)
            print(f"SVM ({kernel}) accuracy: {acc:.4f}")


    # =========================================================
    # 📊 結果統計
    # =========================================================

    print()
    print("########################################################################")
    print("SVM result")
    print()
    print("===== 平均準確率 =====")
    for kernel in kernels:
        mean_acc = np.mean(acc_results[kernel])
        std_acc = np.std(acc_results[kernel])
        print(f"SVM ({kernel}) -> 平均: {mean_acc:.4f}, 標準差: {std_acc:.4f}")


    print("########################################################################")
    print()



"""
