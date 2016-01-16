

[ndata1, text1, alldata1] = xlsread('result_ICIP.xlsx','overlap');


[ndata2, text2, alldata2] = xlsread('result_ICIP.xlsx','cetererror');


ndata=[ndata1; ndata2*100];

ndata=reshape(ndata, [22 24]);

text1(1,2:25)=reshape([text1(1,1:12); text1(1,1:12)], [1,24]);

xlswrite(