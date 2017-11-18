import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
from chainer.dataset import convert
from chainercv.links.model.ssd import multibox_loss


class Updater1(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.dis, self.t_enc = kwargs.pop('models')
        super(Updater1, self).__init__(*args, **kwargs)

    def update_core(self):
        t_enc_optimizer = self.get_optimizer('opt_t_enc')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.dis.xp

        batch_source = self.get_iterator('main').next()
        batch_target = self.get_iterator('target').next()
        batchsize = len(batch_source)
        # x = []
        # for i in range(batchsize):
        #     x.append(np.asarray(batch[i]).astype("f"))
        # x_real = Variable(xp.asarray(x))
        source_fmap = []
        for i in range(6):
            fmap_ = []
            for j in range(batchsize):
                fmap_.append(batch_source[j][i])
            # if len(fmap_) == 1:
            #     fmap_[0] = fmap_[0][np.newaxis,:]
            #     fmap_ =
            # else:
            #     fmap_ = np.vstack(fmap_)
            fmap_ = xp.array(fmap_)
            source_fmap.append(Variable(fmap_))

        y_source = self.dis(source_fmap)

        t_fmap = self.t_enc(Variable(xp.array(batch_target)))
        y_target = self.dis(t_fmap)

        # z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        # x_fake = self.gen(z)
        # y_fake = self.dis(x_fake)

        # loss_dis = F.sum(F.softplus(-y_real)) / batchsize
        # loss_dis += F.sum(F.softplus(y_fake)) / batchsize
        loss_dis = F.sum(F.softplus(-y_source)) / batchsize
        loss_dis += F.sum(F.softplus(y_target)) / batchsize

        loss_t_enc = F.sum(F.softplus(-y_target)) / batchsize

        self.t_enc.cleargrads()
        loss_t_enc.backward()
        t_enc_optimizer.update()
        for map in t_fmap:
            map.unchain_backward()

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_t_enc': loss_t_enc})
        chainer.reporter.report({'loss_dis': loss_dis})

class Updater2(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.dis, self.t_enc = kwargs.pop('models')
        super(Updater2, self).__init__(*args, **kwargs)

    def update_core(self):
        t_enc_optimizer = self.get_optimizer('opt_t_enc')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.dis.xp

        batch_source = self.get_iterator('main').next()
        batch_target = self.get_iterator('target').next()
        batchsize = len(batch_source)
        # x = []
        # for i in range(batchsize):
        #     x.append(np.asarray(batch[i]).astype("f"))
        # x_real = Variable(xp.asarray(x))
        source_fmap = []
        for i in range(1):
            fmap_ = []
            for j in range(batchsize):
                fmap_.append(batch_source[j][i])
            # if len(fmap_) == 1:
            #     fmap_[0] = fmap_[0][np.newaxis,:]
            #     fmap_ =
            # else:
            #     fmap_ = np.vstack(fmap_)
            fmap_ = xp.array(fmap_)
            source_fmap.append(Variable(fmap_))

        y_source = self.dis(source_fmap)

        t_fmap = self.t_enc(Variable(xp.array(batch_target)))
        y_target = self.dis(t_fmap)

        # z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        # x_fake = self.gen(z)
        # y_fake = self.dis(x_fake)

        # loss_dis = F.sum(F.softplus(-y_real)) / batchsize
        # loss_dis += F.sum(F.softplus(y_fake)) / batchsize
        loss_dis = F.sum(F.softplus(-y_source)) / batchsize
        loss_dis += F.sum(F.softplus(y_target)) / batchsize

        loss_t_enc = F.sum(F.softplus(-y_target)) / batchsize

        self.t_enc.cleargrads()
        loss_t_enc.backward()
        t_enc_optimizer.update()
        for map in t_fmap:
            map.unchain_backward()

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_t_enc': loss_t_enc})
        chainer.reporter.report({'loss_dis': loss_dis})

class DA_updater1(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.dis, self.cls = kwargs.pop('models')
        super(DA_updater1, self).__init__(*args, **kwargs)
        self.t_enc = self.cls.extractor
        self.alpha = 1
        self.k = 3

    def update_core(self):
        #t_enc_optimizer = self.get_optimizer('opt_t_enc')
        dis_optimizer = self.get_optimizer('opt_dis')
        cls_optimizer = self.get_optimizer('opt_cls')
        xp = self.dis.xp

        batch_source = self.get_iterator('main').next()
        batch_source_array = convert.concat_examples(batch_source,self.device)
        batch_target = self.get_iterator('target').next()
        batchsize = len(batch_source)

        #compute forwarding with source data
        src_fmap = self.t_enc(batch_source_array[0]) #src feature map
        mb_locs, mb_confs = self.cls.multibox(src_fmap)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
        cls_loss = loc_loss * self.alpha + conf_loss #cls loss

        y_source = self.dis(src_fmap)

        tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))
        y_target = self.dis(tgt_fmap)

        n_fmap_elements = y_target.shape[2]*y_target.shape[3]

        # z = Variable(xp.asarray(self.gen.make_hidden(batchsize)))
        # x_fake = self.gen(z)
        # y_fake = self.dis(x_fake)

        # loss_dis = F.sum(F.softplus(-y_real)) / batchsize
        # loss_dis += F.sum(F.softplus(y_fake)) / batchsize
        loss_dis_src = F.sum(F.softplus(-y_source)) / n_fmap_elements / batchsize
        loss_dis_tgt =  F.sum(F.softplus(y_target)) / n_fmap_elements / batchsize
        #loss_dis = F.sum(F.softplus(-y_source)) / n_fmap_elements / batchsize
        #loss_dis += F.sum(F.softplus(y_target)) / n_fmap_elements / batchsize
        loss_dis = loss_dis_src + loss_dis_tgt

        loss_t_enc = F.sum(F.softplus(-y_target)) / n_fmap_elements / batchsize

        #update cls(and t_enc) by cls_loss and loss_t_enc
        self.cls.cleargrads()
        cls_loss.backward()
        loss_t_enc.backward()
        cls_optimizer.update()
        for s_map, t_map  in zip(src_fmap, tgt_fmap):
             s_map.unchain_backward()
             t_map.unchain_backward()

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_t_enc': loss_t_enc})
        chainer.reporter.report({'loss_dis': loss_dis})
        chainer.reporter.report({'loss_cls': cls_loss})
        chainer.reporter.report({'loss_dis_src': loss_dis_src})
        chainer.reporter.report({'loss_dis_tgt': loss_dis_tgt})

class DA_updater1_buf(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.dis, self.cls = kwargs.pop('models')
        self.buf = kwargs.pop('buffer')
        super(DA_updater1_buf, self).__init__(*args, **kwargs)
        self.t_enc = self.cls.extractor
        self.alpha = 1
        self.k = 3

    def update_core(self):
        #t_enc_optimizer = self.get_optimizer('opt_t_enc')
        dis_optimizer = self.get_optimizer('opt_dis')
        cls_optimizer = self.get_optimizer('opt_cls')
        xp = self.dis.xp

        batch_source = self.get_iterator('main').next()
        batch_source_array = convert.concat_examples(batch_source,self.device)
        batch_target = self.get_iterator('target').next()
        batchsize = len(batch_source)
        use_bufsize = int(batchsize/2)

        #compute forwarding with source data
        src_fmap = self.t_enc(batch_source_array[0]) #src feature map
        mb_locs, mb_confs = self.cls.multibox(src_fmap)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
        cls_loss = loc_loss * self.alpha + conf_loss #cls loss
        self.cls.cleargrads()
        cls_loss.backward()
        cls_loss.unchain_backward()

        # for i in range(len(src_fmap)):
        #     src_fmap[i] = src_fmap[i].data

        func_bGPU = lambda x: chainer.cuda.to_gpu(x, device=self.device) if self.device >= 0 else lambda x: x

        size = 0
        if batchsize >= 2:
            size, e_buf_src , e_buf_tgt = self.buf.get_examples(use_bufsize)
            if size != 0:
                for i in range(len(src_fmap)):
                    #src_fmap[i] = Variable(xp.vstack((src_fmap[i][0:batchsize - size], func_bGPU(e_buf_src[i]))))
                    src_fmap[i] = F.vstack((src_fmap[i][0:batchsize - size], Variable(func_bGPU(e_buf_src[i]))))

        y_source = self.dis(src_fmap)

        #size, e_buf_tgt = self.buf_tgt.get_examples(use_bufsize)
        # if size > 0:
        #     tgt_fmap = self.t_enc(Variable(xp.array(batch_target[0:batchsize-size])))
        #     for i in range(len(tgt_fmap)):
        #         tgt_fmap[i] = F.vstack([tgt_fmap[i],Variable(func_bGPU(e_buf_tgt[i]))])
        # else:
        #     tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))
        tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))
        tgt_fmap_dis = []
        for i in range(len(tgt_fmap)):
            tgt_fmap_dis.append(F.copy(tgt_fmap[i][0:batchsize-size],self.device))
            tgt_fmap_dis[i].unchain_backward()
            if size > 0:
                tgt_fmap_dis[i] = F.vstack([tgt_fmap_dis[i], Variable(func_bGPU(e_buf_tgt[i]))])

        y_target = self.dis(tgt_fmap_dis)
        y_target_enc = self.dis(tgt_fmap)

        n_fmap_elements = y_target.shape[2]*y_target.shape[3]

        loss_dis_src = F.sum(F.softplus(-y_source)) / n_fmap_elements / batchsize
        loss_dis_tgt =  F.sum(F.softplus(y_target)) / n_fmap_elements / batchsize
        loss_dis = loss_dis_src + loss_dis_tgt

        loss_t_enc = F.sum(F.softplus(-y_target_enc)) / n_fmap_elements / batchsize

        #update cls(and t_enc) by cls_loss and loss_t_enc
        loss_t_enc.backward()
        cls_optimizer.update()
        for s_map, t_map  in zip(src_fmap, tgt_fmap):
             s_map.unchain_backward()
             t_map.unchain_backward()

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_t_enc': loss_t_enc})
        chainer.reporter.report({'loss_dis': loss_dis})
        chainer.reporter.report({'loss_cls': cls_loss})
        chainer.reporter.report({'loss_dis_src': loss_dis_src})
        chainer.reporter.report({'loss_dis_tgt': loss_dis_tgt})

        with chainer.no_backprop_mode():
            src_fmap = self.t_enc(batch_source_array[0][-use_bufsize:])
            tgt_fmap = self.t_enc(xp.array(batch_target[-use_bufsize:]))
            for i in range(len(src_fmap)):
                src_fmap[i] = chainer.cuda.to_cpu(src_fmap[i].data)
                tgt_fmap[i] = chainer.cuda.to_cpu(tgt_fmap[i].data)
        self.buf.set_examples(src_fmap,tgt_fmap)

class DA_updater1_buf_multibatch(chainer.training.StandardUpdater):  #hard coding for experiments
    def __init__(self, n_multi_batch = 16, *args, **kwargs):
        self.dis, self.cls = kwargs.pop('models')
        self.buf = kwargs.pop('buffer')
        super(DA_updater1_buf_multibatch, self).__init__(*args, **kwargs)
        self.t_enc = self.cls.extractor
        self.alpha = 1
        self.k = 3
        self.n_multi_batch = n_multi_batch

    def update_core(self):
        #t_enc_optimizer = self.get_optimizer('opt_t_enc')
        dis_optimizer = self.get_optimizer('opt_dis')
        cls_optimizer = self.get_optimizer('opt_cls')
        xp = self.dis.xp

        batch_source = self.get_iterator('main').next()
        batch_source_array = convert.concat_examples(batch_source,self.device)
        #batch_target = self.get_iterator('target').next()
        batchsize = len(batch_source)
        use_bufsize = int(batchsize*self.n_multi_batch / 2) ##

        #compute forwarding with source data
        src_fmap = self.t_enc(batch_source_array[0]) #src feature map
        mb_locs, mb_confs = self.cls.multibox(src_fmap)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
        cls_loss = loc_loss * self.alpha + conf_loss #cls loss
        self.cls.cleargrads()
        cls_loss.backward()
        cls_loss.unchain_backward()

        # for i in range(len(src_fmap)):
        #     src_fmap[i] = src_fmap[i].data

        func_bGPU = lambda x: chainer.cuda.to_gpu(x, device=self.device) if self.device >= 0 else lambda x: x

        size = 0
        if batchsize >= 2:
            size, e_buf_src , e_buf_tgt = self.buf.get_examples(use_bufsize)
            # if size != 0:
            #     for i in range(len(src_fmap)):
            #         #src_fmap[i] = Variable(xp.vstack((src_fmap[i][0:batchsize - size], func_bGPU(e_buf_src[i]))))
            #         src_fmap[i] = F.vstack((src_fmap[i][0:batchsize - size], Variable(func_bGPU(e_buf_src[i]))))
        bufsize_split = int(size/self.n_multi_batch )

        cls_loss_report = cls_loss.data
        loss_t_enc_report = 0
        loss_dis_report = 0
        loss_dis_src_report = 0
        loss_dis_tgt_report = 0

        for i in range(self.n_multi_batch ):
            if i != 0:
                batch_source = self.get_iterator('main').next()
                batch_source_array = convert.concat_examples(batch_source, self.device)
                with chainer.no_backprop_mode():
                    src_fmap = self.t_enc(batch_source_array[0])
            if bufsize_split > 0:
                for j in range(len(src_fmap)):
                    src_fmap[j] = F.vstack((src_fmap[j][0:batchsize - bufsize_split], Variable(func_bGPU(e_buf_src[j][bufsize_split*i:bufsize_split*(i+1)]))))
            y_source = self.dis(src_fmap)

        #size, e_buf_tgt = self.buf_tgt.get_examples(use_bufsize)
        # if size > 0:
        #     tgt_fmap = self.t_enc(Variable(xp.array(batch_target[0:batchsize-size])))
        #     for i in range(len(tgt_fmap)):
        #         tgt_fmap[i] = F.vstack([tgt_fmap[i],Variable(func_bGPU(e_buf_tgt[i]))])
        # else:
        #     tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))
            batch_target = self.get_iterator('target').next()
            tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))
            tgt_fmap_dis = []
            for j in range(len(tgt_fmap)):
                tgt_fmap_dis.append(F.copy(tgt_fmap[j][0:batchsize-bufsize_split],self.device))
                tgt_fmap_dis[j].unchain_backward()
                if bufsize_split > 0:
                    tgt_fmap_dis[j] = F.vstack([tgt_fmap_dis[j], Variable(func_bGPU(e_buf_tgt[j][bufsize_split*i:bufsize_split*(i+1)]))])

            y_target = self.dis(tgt_fmap_dis)
            y_target_enc = self.dis(tgt_fmap)

            n_fmap_elements = y_target.shape[2]*y_target.shape[3]

            loss_dis_src = F.sum(F.softplus(-y_source)) / n_fmap_elements / batchsize / self.n_multi_batch
            loss_dis_tgt =  F.sum(F.softplus(y_target)) / n_fmap_elements / batchsize / self.n_multi_batch
            loss_dis = loss_dis_src + loss_dis_tgt

            loss_t_enc = F.sum(F.softplus(-y_target_enc)) / n_fmap_elements / batchsize / self.n_multi_batch

            #update cls(and t_enc) by cls_loss and loss_t_enc
            loss_t_enc.backward()
            #cls_optimizer.update()
            for s_map, t_map  in zip(src_fmap, tgt_fmap):
                 s_map.unchain_backward()
                 t_map.unchain_backward()

            self.dis.cleargrads()
            loss_dis.backward()
            #dis_optimizer.update()

            loss_t_enc_report += loss_t_enc.data
            loss_dis_report += loss_dis.data
            loss_dis_src_report += loss_dis_src.data
            loss_dis_tgt_report += loss_dis_tgt.data

            with chainer.no_backprop_mode():
                src_fmap = self.t_enc(batch_source_array[0][-bufsize_split:])
                tgt_fmap = self.t_enc(xp.array(batch_target[-bufsize_split:]))
                for j in range(len(src_fmap)):
                    src_fmap[j] = chainer.cuda.to_cpu(src_fmap[j].data)
                    tgt_fmap[j] = chainer.cuda.to_cpu(tgt_fmap[j].data)
            if i == 0:
                buf_src_fmap = src_fmap
                buf_tgt_fmap = tgt_fmap
            else:
                for j in range(len(src_fmap)):
                    buf_src_fmap[j] = np.vstack((buf_src_fmap[j],src_fmap[j]))
                    buf_tgt_fmap[j] = np.vstack((buf_tgt_fmap[j], tgt_fmap[j]))

        cls_optimizer.update()
        dis_optimizer.update()

        chainer.reporter.report({'loss_t_enc': loss_t_enc_report})
        chainer.reporter.report({'loss_dis': loss_dis_report})
        chainer.reporter.report({'loss_cls': cls_loss_report})
        chainer.reporter.report({'loss_dis_src': loss_dis_src_report})
        chainer.reporter.report({'loss_dis_tgt': loss_dis_tgt_report})

        self.buf.set_examples(buf_src_fmap,buf_tgt_fmap)

class DA_updater1_buf_2(chainer.training.StandardUpdater):
    def __init__(self, bufmode = 0,batchmode = 0, cls_train_mode = 0, *args, **kwargs):
        self.dis, self.cls = kwargs.pop('models')
        self.buf = kwargs.pop('buffer')
        super(DA_updater1_buf_2, self).__init__(*args, **kwargs)
        self.t_enc = self.cls.extractor
        self.alpha = 1
        self.k = 3
        self.bufmode = bufmode
        self.batchmode = batchmode
        self.cls_train_mode = cls_train_mode

    def update_core(self):
        #t_enc_optimizer = self.get_optimizer('opt_t_enc')
        dis_optimizer = self.get_optimizer('opt_dis')
        cls_optimizer = self.get_optimizer('opt_cls')
        xp = self.dis.xp

        batch_source = self.get_iterator('main').next()
        batch_source_array = convert.concat_examples(batch_source,self.device)
        batch_target = self.get_iterator('target').next()
        batchsize = len(batch_source)
        use_bufsize = int(batchsize/2)

        #train discriminator
        src_fmap = self.t_enc(batch_source_array[0]) #src feature map

        # mb_locs, mb_confs = self.cls.multibox(src_fmap)
        # loc_loss, conf_loss = multibox_loss(
        #     mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
        # cls_loss = loc_loss * self.alpha + conf_loss #cls loss
        # self.cls.cleargrads()
        # cls_loss.backward()
        # cls_loss.unchain_backward()

        # for i in range(len(src_fmap)):
        #     src_fmap[i] = src_fmap[i].data

        func_bGPU = lambda x: chainer.cuda.to_gpu(x, device=self.device) if self.device >= 0 else lambda x: x

        size = 0
        if batchsize >= 2:
            size, e_buf_src , e_buf_tgt = self.buf.get_examples(use_bufsize)
        if size != 0:
            src_fmap_dis = []
            for i in range(len(src_fmap)):
                #src_fmap[i] = Variable(xp.vstack((src_fmap[i][0:batchsize - size], func_bGPU(e_buf_src[i]))))
                src_fmap_dis.append(F.vstack((F.copy(src_fmap[i][0:batchsize - size]), Variable(func_bGPU(e_buf_src[i])))))
                src_fmap_dis[i].unchain_backward()
        else:
            src_fmap_dis = []
            for i in range(len(src_fmap)):
                src_fmap_dis.append(F.copy(src_fmap[i]))
                src_fmap_dis[i].unchain_backward()

        y_source = self.dis(src_fmap_dis)

        #size, e_buf_tgt = self.buf_tgt.get_examples(use_bufsize)
        # if size > 0:
        #     tgt_fmap = self.t_enc(Variable(xp.array(batch_target[0:batchsize-size])))
        #     for i in range(len(tgt_fmap)):
        #         tgt_fmap[i] = F.vstack([tgt_fmap[i],Variable(func_bGPU(e_buf_tgt[i]))])
        # else:
        #     tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))
        tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))
        tgt_fmap_dis = []
        for i in range(len(tgt_fmap)):
            tgt_fmap_dis.append(F.copy(tgt_fmap[i][0:batchsize-size],self.device))
            tgt_fmap_dis[i].unchain_backward()
            if size > 0:
                tgt_fmap_dis[i] = F.vstack([tgt_fmap_dis[i], Variable(func_bGPU(e_buf_tgt[i]))])

        y_target = self.dis(tgt_fmap_dis)
        # y_target_enc = self.dis(tgt_fmap)

        n_fmap_elements = y_target.shape[2]*y_target.shape[3]

        loss_dis_src = F.sum(F.softplus(-y_source)) / n_fmap_elements / batchsize
        loss_dis_tgt =  F.sum(F.softplus(y_target)) / n_fmap_elements / batchsize
        loss_dis = loss_dis_src + loss_dis_tgt

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        #save fmap to buffer
        if self.bufmode == 1:
            src_fmap_tobuf = []
            tgt_fmap_tobuf = []
            for i in range(len(src_fmap)):
                src_fmap_tobuf.append(chainer.cuda.to_cpu(src_fmap[i].data[:use_bufsize]))
                tgt_fmap_tobuf.append(chainer.cuda.to_cpu(tgt_fmap[i].data[:use_bufsize]))
            self.buf.set_examples(src_fmap_tobuf, tgt_fmap_tobuf)

        if self.batchmode == 1:
            y_target_enc = self.dis(tgt_fmap)
            mb_locs, mb_confs = self.cls.multibox(src_fmap)
            loc_loss, conf_loss = multibox_loss(
                mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
            cls_loss = loc_loss * self.alpha + conf_loss  # cls loss

        if self.bufmode == 0 or self.batchmode == 0:
            batch_source = self.get_iterator('main').next()
            batch_source_array = convert.concat_examples(batch_source, self.device)
            batch_target = self.get_iterator('target').next()
            src_fmap = self.t_enc(batch_source_array[0])  # src feature map
            tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))

        if self.batchmode == 0:
            y_target_enc = self.dis(tgt_fmap)
        loss_t_enc = F.sum(F.softplus(-y_target_enc)) / n_fmap_elements / batchsize

        #update cls(and t_enc) by cls_loss and loss_t_enc
        self.cls.cleargrads()
        loss_t_enc.backward()
        if self.cls_train_mode == 1:
            cls_optimizer.update()

        if self.batchmode == 0:
            mb_locs, mb_confs = self.cls.multibox(src_fmap)
            loc_loss, conf_loss = multibox_loss(
                mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
            cls_loss = loc_loss * self.alpha + conf_loss #cls loss

        if self.cls_train_mode == 1:
            self.cls.cleargrads()

        cls_loss.backward()
        cls_optimizer.update()

        if self.bufmode == 0:
            src_fmap_tobuf = []
            tgt_fmap_tobuf = []
            for i in range(len(src_fmap)):
                src_fmap_tobuf.append(chainer.cuda.to_cpu(src_fmap[i].data[:use_bufsize]))
                tgt_fmap_tobuf.append(chainer.cuda.to_cpu(tgt_fmap[i].data[:use_bufsize]))
            self.buf.set_examples(src_fmap_tobuf, tgt_fmap_tobuf)

        for s_map, t_map  in zip(src_fmap, tgt_fmap):
             s_map.unchain_backward()
             t_map.unchain_backward()

        chainer.reporter.report({'loss_t_enc': loss_t_enc})
        chainer.reporter.report({'loss_dis': loss_dis})
        chainer.reporter.report({'loss_cls': cls_loss})
        chainer.reporter.report({'loss_dis_src': loss_dis_src})
        chainer.reporter.report({'loss_dis_tgt': loss_dis_tgt})

        # with chainer.no_backprop_mode():
        #     src_fmap = self.t_enc(batch_source_array[0][-use_bufsize:])
        #     tgt_fmap = self.t_enc(xp.array(batch_target[-use_bufsize:]))
        #     for i in range(len(src_fmap)):
        #         src_fmap[i] = chainer.cuda.to_cpu(src_fmap[i].data)
        #         tgt_fmap[i] = chainer.cuda.to_cpu(tgt_fmap[i].data)
        # self.buf.set_examples(src_fmap,tgt_fmap)



