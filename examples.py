import os
import numpy as np
import processing as p
from multiprocessing import Process, Queue

def main(catalog,cond):

    num_procs = 15
    split = np.array_split(catalog[cond], num_procs)

    def slave(queue, chunk):
        for entry in chunk:
            y = p.WISP_Source(catalog=catalog,entry=entry,data_dir=data_dir,output_dir='.')
            y.process()
            pars = y.get_subpx_pars()
            items = (entry['NUMBER'],pars)
            queue.put(items)
        queue.put(None)

    queue = Queue()
    procs = [Process(target=slave, args=(queue,chunk)) for chunk in split]
    for proc in procs: proc.start()

    res = np.zeros((len(catalog[cond]),100)) * np.NaN
    finished,i = 0,0
    while finished < num_procs:
        items = queue.get()
        if items == None:
            finished += 1
        else:
            obj_num, pars = items
            res[i,0] = obj_num
            for j,x in enumerate(pars): res[i,j+1] = x
            i+=1

    for proc in procs: proc.join()

    asort = np.argsort(res[:,0])
    res = res[asort,:]
    return res

if __name__ == '__main__':

    data_dir='/data/highzgal/PUBLICACCESS/WISPS/data/V5.0/'

    catalog = p.WISP_Catalog(par_num=167,grism='G141',data_dir=data_dir).get_catalog()
    cond1 = (catalog.NUMBER < 1000)
    cond2 = (20 <= catalog.MAG) & (catalog.MAG <= 23.5)
    cond3 = (catalog.CLASS_STAR < 0.1)
    cond4 = np.array([os.path.isfile('%s/Par%s/G102_DRIZZLE/aXeWFC3_G102_mef_ID%i.fits' % (data_dir,catalog['PAR_NUM'][0],i)) for i in catalog['NUMBER']])
    cond5 = np.array([os.path.isfile('%s/Par%s/G141_DRIZZLE/aXeWFC3_G141_mef_ID%i.fits' % (data_dir,catalog['PAR_NUM'][0],i)) for i in catalog['NUMBER']])
    cond  = cond1 & cond2 & cond3 & cond4 & cond5
    short_list = catalog[cond]
    print short_list.NUMBER

    c = p.WISP_Catalog(par_num=167,grism='G102',data_dir=data_dir).get_catalog()
    f = p.WISP_Field(data_dir=data_dir,output_dir='.',catalog=c,background=0)
    #f.process()
    s = p.WISP_Source(catalog=c,entry=c[7],data_dir=data_dir,output_dir='.')
    s.process()
    #res = main(c,cond)
    #np.savetxt('profile_G102_pars.dat',res,fmt='%5.f '+' '.join(['%8.2e' for i in range(res.shape[1]-1)]))

    c = p.WISP_Catalog(par_num=167,grism='G141',data_dir=data_dir).get_catalog()
    f = p.WISP_Field(data_dir=data_dir,output_dir='.',catalog=c,background=0)
    #f.process()
    s = p.WISP_Source(catalog=c,entry=c[7],data_dir=data_dir,output_dir='.')
    s.process()
    #res = main(c,cond)
    #np.savetxt('profile_G141_pars.dat',res,fmt='%5.f '+' '.join(['%8.2e' for i in range(res.shape[1]-1)]))