PHAN LOAI HOA IRIS BANG DECISION TREE
Do an mon Khai Pha Du Lieu - Truong Dai Hoc Sai Gon

MO TA CAC TAP TIN TRONG THU MUC

1. main.py
   Chuong trinh chinh. Dung DecisionTreeClassifier (scikit-learn) de phan loai
   150 mau hoa Iris thanh 3 loai (setosa, versicolor, virginica).
   Danh gia bang stratified 10-fold cross validation, in ra: AUC, CA, F1,
   Precision, Recall, MCC, ma tran nham lan va cay quyet dinh.
   Du lieu duoc tai truc tiep tu UCI (id = 53) bang thu vien ucimlrepo.

2. iris.csv
   Du lieu Iris (150 dong, 4 dac trung + 1 nhan), tao tu iris.data goc cua UCI.
   Dung khi chay offline: trong main.py bo comment khoi "Cach 2" va comment
   khoi fetch_ucirepo lai.

3. BaoCao_DecisionTree_Iris.pdf
   Bao cao day du (25 trang, 6 chuong): gioi thieu, mo ta du lieu, phuong phap,
   thuc nghiem, ket qua (so sanh voi Orange Data Mining), thao luan va ket luan.

4. readme.txt
   Tap tin nay.

CACH CHAY

   pip install ucimlrepo pandas scikit-learn
   python main.py

KET QUA (10-fold cross validation)

   AUC 0.963 | CA 0.940 | F1 0.940 | Precision 0.940 | Recall 0.940 | MCC 0.910
