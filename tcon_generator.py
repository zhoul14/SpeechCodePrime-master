def tcon_generate(tconname, trainiter,fdim, triphone, filenum, filename, initCBname, outCBname, dataname, idxname, tagname,isoword = 0, usecuda = 0,segment = 0,dureweight = 5):
    f = open(tconname, 'w')
    f.write('PARAM:\n')
    f.write('TRAINITER %d\n'%trainiter)
    if triphone and isoword
        f.write('DICTCONFIG  D:/MyCodes/DDBHMMTasks/Didict/worddict.txt\n')
    elif triphone
        f.write('DICTCONFIG  D:/MyCodes/DDBHMMTasks/Didict/worddict2.txt\n')
    elif
        f.write('DICTCONFIG  D:/MyCodes/DDBHMMTasks/dict/worddict.txt\n')
    f.write('TRAINNUM %d\n'%filenum)
    f.write('EMITER 10\n')
    f.write('TRIPHONE %d\n'%triphone)
    f.write('FDIM %d\n'%fdim)
    f.write('DURWEIGHT %d\n'%DURWEIGHT)
    f.write('INITCODEBOOK %s'%initCBname%fdim)
    f.write('USECUDA %d'%usecuda)
    f.write('OUTCODEBOOK %s'%outCBname)
    f.write('COEF 1')
    f.write('OUTCODEBOOK %s'%outCBname)
    f.write('DATA:\n')
    for i in range(filenum):
        f.write('"E:/Speech/isoword/male/d%d2/M%.2d.d%d2 %s %s\n'%(fdim,i + 1,idxname,tagname, i + 1))