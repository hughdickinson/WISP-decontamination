import os
import numpy as np
import processing as p
from multiprocessing import Process, Queue

def main(short_list,grism):

    num_procs = 15
    split = np.array_split(short_list, num_procs)

    def slave(queue, chunk):
        for entry in chunk:
            y = p.WISP_Source(par_num=entry['PAR_NUM'],obj_num=entry['NUMBER'],grism=grism,data_dir=data_dir,output_dir=output_dir)
            y.process()
            pars = y.get_subpx_pars()
            items = (entry['NUMBER'],pars)
            queue.put(items)
        queue.put(None)

    queue = Queue()
    procs = [Process(target=slave, args=(queue,chunk)) for chunk in split]
    for proc in procs: proc.start()

    res = np.zeros((len(short_list),100)) * np.NaN
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

    data_dir = '/data/highzgal/PUBLICACCESS/WISPS/data/V5.0/'
    output_dir = '/data/highzgal/mehta/WISP/WISP-decontamination/output/'

    catalog = p.WISP_Catalog(par_num=167,grism='G141',data_dir=data_dir).get_catalog()
    cond1 = (catalog.NUMBER < 1000)
    cond2 = (20 <= catalog.MAG) & (catalog.MAG <= 23.5)
    cond3 = (catalog.CLASS_STAR < 0.1)
    cond4 = np.array([os.path.isfile('%s/Par%s/G102_DRIZZLE/aXeWFC3_G102_mef_ID%i.fits' % (data_dir,catalog['PAR_NUM'][0],i)) for i in catalog['NUMBER']])
    cond5 = np.array([os.path.isfile('%s/Par%s/G141_DRIZZLE/aXeWFC3_G141_mef_ID%i.fits' % (data_dir,catalog['PAR_NUM'][0],i)) for i in catalog['NUMBER']])
    cond  = cond1 & cond2 & cond3 & cond4 & cond5
    short_list = catalog[cond]
    print short_list.NUMBER

    for grism in ['G102','G141']:
        s = p.WISP_Source(par_num=167,obj_num=8,grism=grism,data_dir=data_dir,output_dir=output_dir)
        s.process()
        #res = main(short_list,grism)
        #np.savetxt(output_dir+'Par167/profile_%s_pars.dat'%grism,res,fmt='%5.f '+' '.join(['%8.2e' for i in range(res.shape[1]-1)]))
