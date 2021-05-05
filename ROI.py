from operator import add, truediv

def cost(train, test):
    x_f = 1
    x_l = 0.5
    x_t = 0.1

    C_fixed = x_f/60
    C_label = x_l/60
    C_tuning = x_t/60
    H = 1
    C_resource = 400

    # train_size = [300,400,500]
    # test_size = [200,200,200]
    
    train_size = train
    test_size = test
    
    train_cost = []
    test_cost = []

    for i in train_size :
        t = (C_fixed + C_label + C_tuning) * i
        train_cost.append(t)
        
        
    print("train cost = " + str(train_cost))

    for b in test_size:
        te = (C_fixed) * b
        test_cost.append(te)

    print("test cost = " + str(test_cost))

    # using map() + add() to 
    # add two list 
    cost = list(map(add, train_cost, test_cost))
    print("cost = " + str(cost))
    
    Total_cost = []
    
    for a in cost:
        tc = a * H * C_resource
        Total_cost.append(tc)
     
    print("Total cost = " + str(Total_cost))
    return Total_cost


def benefit(TP,FN):
    reward = 500
    penalty = 500
    
    tp = TP
    fn = FN
    
    tp_correct = []
    fn_incorrect = []
    
    for i in tp:
        result = reward * i
        tp_correct.append(result)
    
    print("classified correctly = " + str(tp_correct))
    
    for b in fn:
        result = penalty * b
        fn_incorrect.append(result)
    
    print("classified incorrectly = " + str(fn_incorrect))
    
    benefits = []
    zip_list = zip(tp_correct, fn_incorrect)
    
    for list1_tp, list2_fn in zip_list:
        benefits.append(list1_tp - list2_fn)
       
    print("benefit = " + str(benefits))
    return benefits
 


def roi():
    # typo 3
    # TP = [152,166,170,175,177,172,181,186,191,193,196,208,217,221]
    # FN = [122,108,104,99,97,102,93,88,83,81,78,66,57,53]  
    
    # train_size = [150,300,450,600,750,900,1050,1200,1350,1500,1650,1800,1950,2150]
    # test_size = [547,547,547,547,547,547,547,547,547,547,547,547,547,547]
    
    # redmine
    TP = [289,293,346,346,380,391,389,381,418,416,421,426,428,443,456]
    FN = [273,269,216,216,182,171,173,181,144,146,141,136,134,119,106]  
    
    train_size = [300,600,900,1200,1500,1800,2100,2400,2700,3000,3300,3600,3900,4200,4450]
    test_size = [1122,1122,1122,1122,1122,1122,1122,1122,1122,1122,1122,1122,1122,1122,1122]
    
    # ruby
    # TP = [113,113,116,111,112,108,107,118,127,122,124,128,127,133,132]
    # FN = [67,67,64,69,68,72,73,62,53,58,56,52,53,47,48]  
    
    # train_size = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1420]
    # test_size = [361,361,361,361,361,361,361,361,361,361,361,361,361,361,361]
     
    benefits = benefit(TP, FN)
    costs = cost(train_size, test_size)
    
    differnce = []
    zip_list = zip(benefits, costs)
    
    for list1_b, list2_c in zip_list:
        differnce.append(list1_b - list2_c)
    
    print("benefit - cost = " + str(differnce))
    # division of lists
    # using map()
    result = list(map(truediv, differnce, costs))
    
    print("ROI for redmine = " + str(result))

roi()