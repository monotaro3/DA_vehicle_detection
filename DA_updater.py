import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
from chainer.dataset import convert
from chainercv.links.model.ssd import multibox_loss
import six
from chainer.backends import cuda


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

        func_bGPU = (lambda x: chainer.cuda.to_gpu(x, device=self.device)) if self.device >= 0 else lambda x: x

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

    def serialize(self, serializer):
        # """Serializes the current state of the updater object."""
        # for name, iterator in six.iteritems(self._iterators):
        #     iterator.serialize(serializer['iterator:' + name])
        #
        # for name, optimizer in six.iteritems(self._optimizers):
        #     optimizer.serialize(serializer['optimizer:' + name])
        #     optimizer.target.serialize(serializer['model:' + name])
        #
        # self.iteration = serializer('iteration', self.iteration)
        super().serialize(serializer)

        self.buf.serialize(serializer['buf'])

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

        func_bGPU = (lambda x: chainer.cuda.to_gpu(x, device=self.device)) if self.device >= 0 else lambda x: x

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
    def __init__(self, bufmode = 0,batchmode = 0, cls_train_mode = 0, init_disstep = 1, init_tgtstep = 1, tgt_steps_schedule = None, *args, **kwargs):
        self.dis, self.cls = kwargs.pop('models')
        self.buf = kwargs.pop('buffer')
        super(DA_updater1_buf_2, self).__init__(*args, **kwargs)
        self.t_enc = self.cls.extractor
        self.alpha = 1
        self.k = 3
        self.bufmode = bufmode
        self.batchmode = batchmode
        self.cls_train_mode = cls_train_mode
        self.current_dis_step = init_disstep
        self.current_tgt_step = init_tgtstep
        self.tgt_steps_schedule = tgt_steps_schedule
        if self.tgt_steps_schedule != None:
            if isinstance(tgt_steps_schedule, list):
                self.tgt_steps_schedule.sort(key=lambda x:x[0])
            else:
                print("tgt step schedule must be specified by list object. The schedule is ignored.")
                self.tgt_steps_schedule = None

    def update_core(self):
        if isinstance(self.tgt_steps_schedule,list) and len(self.tgt_steps_schedule) > 0:
            while len(self.tgt_steps_schedule) > 0 and self.tgt_steps_schedule[0][0] < self.iteration:
                self.tgt_steps_schedule.pop(0)
            if len(self.tgt_steps_schedule) > 0:
                if self.tgt_steps_schedule[0][0] == self.iteration:
                    self.current_tgt_step = self.tgt_steps_schedule[0][1]
                    self.tgt_steps_schedule.pop(0)

        #t_enc_optimizer = self.get_optimizer('opt_t_enc')
        dis_optimizer = self.get_optimizer('opt_dis')
        cls_optimizer = self.get_optimizer('opt_cls')
        xp = self.dis.xp
        func_bGPU = (lambda x: chainer.cuda.to_gpu(x, device=self.device)) if self.device >= 0 else lambda x: x

        loss_dis_src_sum = 0
        loss_dis_tgt_sum = 0
        loss_dis_sum = 0

        try:
            src_fmaps_iter = self.get_iterator('src_fmaps')
            fix_src_encoder = True
        except KeyError:
            fix_src_encoder = False

        for z in range(self.current_dis_step):
            if not fix_src_encoder or (z == self.current_dis_step -1 and (self.bufmode ==1 or self.batchmode == 1)):
                batch_source = self.get_iterator('main').next()
                batch_source_array = convert.concat_examples(batch_source,self.device)
                src_fmap = self.t_enc(batch_source_array[0])  # src feature map
            batch_target = self.get_iterator('target').next()
            batchsize = len(batch_target)
            use_bufsize = int(batchsize/2)

            #train discriminator


            # mb_locs, mb_confs = self.cls.multibox(src_fmap)
            # loc_loss, conf_loss = multibox_loss(
            #     mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
            # cls_loss = loc_loss * self.alpha + conf_loss #cls loss
            # self.cls.cleargrads()
            # cls_loss.backward()
            # cls_loss.unchain_backward()

            # for i in range(len(src_fmap)):
            #     src_fmap[i] = src_fmap[i].data

            size = 0
            if batchsize >= 2:
                size, e_buf_src , e_buf_tgt = self.buf.get_examples(use_bufsize)
            if fix_src_encoder:
                src_fmap_dis = []
                batch_source_fixed = src_fmaps_iter.next()
                batchsize_fixed = len(batch_source_fixed)
                # batch_source_array_fixed = convert.concat_examples(batch_source_fixed, self.device)
                for i in range(len(batch_source_fixed[0])):
                    fmap_ = []
                    for j in range(batchsize_fixed):
                        fmap_.append(batch_source_fixed[j][i])
                    # if len(fmap_) == 1:
                    #     fmap_[0] = fmap_[0][np.newaxis,:]
                    #     fmap_ =
                    # else:
                    #     fmap_ = np.vstack(fmap_)
                    fmap_ = xp.array(fmap_)
                    src_fmap_dis.append(Variable(fmap_))
            else:
                if size != 0:
                    src_fmap_dis = []
                    for i in range(len(src_fmap)):
                        #src_fmap[i] = Variable(xp.vstack((src_fmap[i][0:batchsize - size], func_bGPU(e_buf_src[i]))))
                        src_fmap_dis.append(F.vstack((F.copy(src_fmap[i][0:batchsize - size],self.device), Variable(func_bGPU(e_buf_src[i])))))
                        src_fmap_dis[i].unchain_backward()
                else:
                    src_fmap_dis = []
                    for i in range(len(src_fmap)):
                        src_fmap_dis.append(F.copy(src_fmap[i],self.device))
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

            loss_dis_src_sum += loss_dis_src.data
            loss_dis_tgt_sum += loss_dis_tgt.data
            loss_dis_sum += loss_dis.data

            self.dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()

        loss_dis_src_sum /= self.current_dis_step
        loss_dis_tgt_sum /= self.current_dis_step
        loss_dis_sum /= self.current_dis_step

        loss_t_enc_sum = 0
        loss_cls_sum = 0

        for i in range(self.current_tgt_step):
            #save fmap to buffer
            if i == 0 and self.bufmode == 1:
                src_fmap_tobuf = []
                tgt_fmap_tobuf = []
                for i in range(len(src_fmap)):
                    src_fmap_tobuf.append(chainer.cuda.to_cpu(src_fmap[i].data[:use_bufsize]))
                    tgt_fmap_tobuf.append(chainer.cuda.to_cpu(tgt_fmap[i].data[:use_bufsize]))
                self.buf.set_examples(src_fmap_tobuf, tgt_fmap_tobuf)

            if i == 0 and self.batchmode == 1:
                y_target_enc = self.dis(tgt_fmap)
                mb_locs, mb_confs = self.cls.multibox(src_fmap)
                loc_loss, conf_loss = multibox_loss(
                    mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
                cls_loss = loc_loss * self.alpha + conf_loss  # cls loss

            if i > 0 or self.bufmode == 0 or self.batchmode == 0:
                batch_source = self.get_iterator('main').next()
                batch_source_array = convert.concat_examples(batch_source, self.device)
                batch_target = self.get_iterator('target').next()
                src_fmap = self.t_enc(batch_source_array[0])  # src feature map
                tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))

            if i > 0 or self.batchmode == 0:
                y_target_enc = self.dis(tgt_fmap)
            loss_t_enc = F.sum(F.softplus(-y_target_enc)) / n_fmap_elements / batchsize

            #update cls(and t_enc) by cls_loss and loss_t_enc
            self.cls.cleargrads()
            loss_t_enc.backward()
            if self.cls_train_mode == 1:
                cls_optimizer.update()

            if i > 0 or self.batchmode == 0:
                mb_locs, mb_confs = self.cls.multibox(src_fmap)
                loc_loss, conf_loss = multibox_loss(
                    mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
                cls_loss = loc_loss * self.alpha + conf_loss #cls loss

            if self.cls_train_mode == 1:
                self.cls.cleargrads()

            cls_loss.backward()
            cls_optimizer.update()

            if i == 0 and self.bufmode == 0:
                src_fmap_tobuf = []
                tgt_fmap_tobuf = []
                for i in range(len(src_fmap)):
                    src_fmap_tobuf.append(chainer.cuda.to_cpu(src_fmap[i].data[:use_bufsize]))
                    tgt_fmap_tobuf.append(chainer.cuda.to_cpu(tgt_fmap[i].data[:use_bufsize]))
                self.buf.set_examples(src_fmap_tobuf, tgt_fmap_tobuf)

            for s_map, t_map  in zip(src_fmap, tgt_fmap):
                 s_map.unchain_backward()
                 t_map.unchain_backward()

            loss_t_enc_sum += loss_t_enc.data
            loss_cls_sum += cls_loss.data

        loss_t_enc_sum /= self.current_tgt_step
        loss_cls_sum /= self.current_tgt_step

        chainer.reporter.report({'loss_t_enc': loss_t_enc_sum})
        chainer.reporter.report({'loss_dis': loss_dis_sum})
        chainer.reporter.report({'loss_cls': loss_cls_sum})
        chainer.reporter.report({'loss_dis_src': loss_dis_src_sum})
        chainer.reporter.report({'loss_dis_tgt': loss_dis_tgt_sum})

        # with chainer.no_backprop_mode():
        #     src_fmap = self.t_enc(batch_source_array[0][-use_bufsize:])
        #     tgt_fmap = self.t_enc(xp.array(batch_target[-use_bufsize:]))
        #     for i in range(len(src_fmap)):
        #         src_fmap[i] = chainer.cuda.to_cpu(src_fmap[i].data)
        #         tgt_fmap[i] = chainer.cuda.to_cpu(tgt_fmap[i].data)
        # self.buf.set_examples(src_fmap,tgt_fmap)

class DA_updater1_buf_2_coral(chainer.training.StandardUpdater):
    def __init__(self, bufmode = 1,batchmode = 0, cls_train_mode = 0, init_disstep = 1, init_tgtstep = 1, tgt_steps_schedule = None, *args, **kwargs):
        self.dis, self.cls = kwargs.pop('models')
        self.buf = kwargs.pop('buffer')
        # self.coral_loss_func = kwargs.pop('coral_loss_func')
        self.coral_batchsize = kwargs.pop('coral_batchsize')
        self.CORAL_weight = kwargs.pop('coral_weight')
        self.gpu_num = kwargs["device"]
        super(DA_updater1_buf_2_coral, self).__init__(*args, **kwargs)
        self.t_enc = self.cls.extractor
        self.alpha = 1
        self.k = 3
        self.bufmode = bufmode
        self.batchmode = batchmode
        self.cls_train_mode = cls_train_mode
        self.current_dis_step = init_disstep
        self.current_tgt_step = init_tgtstep
        self.tgt_steps_schedule = tgt_steps_schedule

        if self.tgt_steps_schedule != None:
            if isinstance(tgt_steps_schedule, list):
                self.tgt_steps_schedule.sort(key=lambda x:x[0])
            else:
                print("tgt step schedule must be specified by list object. The schedule is ignored.")
                self.tgt_steps_schedule = None

    def update_core(self):
        if isinstance(self.tgt_steps_schedule,list) and len(self.tgt_steps_schedule) > 0:
            while len(self.tgt_steps_schedule) > 0 and self.tgt_steps_schedule[0][0] < self.iteration:
                self.tgt_steps_schedule.pop(0)
            if len(self.tgt_steps_schedule) > 0:
                if self.tgt_steps_schedule[0][0] == self.iteration:
                    self.current_tgt_step = self.tgt_steps_schedule[0][1]
                    self.tgt_steps_schedule.pop(0)

        #t_enc_optimizer = self.get_optimizer('opt_t_enc')
        dis_optimizer = self.get_optimizer('opt_dis')
        cls_optimizer = self.get_optimizer('opt_cls')
        xp = self.dis.xp
        func_bGPU = (lambda x: chainer.cuda.to_gpu(x, device=self.gpu_num)) if isinstance(self.device, cuda.GpuDevice) else lambda x: x

        loss_dis_src_sum = 0
        loss_dis_tgt_sum = 0
        loss_dis_sum = 0
        loss_coral_sum = 0

        try:
            src_fmaps_iter = self.get_iterator('src_fmaps')
            fix_src_encoder = True
        except KeyError:
            fix_src_encoder = False

        for z in range(self.current_dis_step):
            if not fix_src_encoder or (z == self.current_dis_step -1 and (self.bufmode ==1 or self.batchmode == 1)):
                batch_source = self.get_iterator('main').next()
                batch_source_array = convert.concat_examples(batch_source,self.device)
                src_fmap = self.t_enc(batch_source_array[0])  # src feature map
            batch_target = self.get_iterator('target').next()
            batchsize = len(batch_target)
            use_bufsize = int(batchsize/2)

            #train discriminator


            # mb_locs, mb_confs = self.cls.multibox(src_fmap)
            # loc_loss, conf_loss = multibox_loss(
            #     mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
            # cls_loss = loc_loss * self.alpha + conf_loss #cls loss
            # self.cls.cleargrads()
            # cls_loss.backward()
            # cls_loss.unchain_backward()

            # for i in range(len(src_fmap)):
            #     src_fmap[i] = src_fmap[i].data

            size = 0
            if batchsize >= 2:
                size, e_buf_src , e_buf_tgt = self.buf.get_examples(use_bufsize)
            if fix_src_encoder:
                pass
            #     src_fmap_dis = []
            #     batch_source_fixed = src_fmaps_iter.next()
            #     batchsize_fixed = len(batch_source_fixed)
            #     # batch_source_array_fixed = convert.concat_examples(batch_source_fixed, self.device)
            #     for i in range(len(batch_source_fixed[0])):
            #         fmap_ = []
            #         for j in range(batchsize_fixed):
            #             fmap_.append(batch_source_fixed[j][i])
            #         # if len(fmap_) == 1:
            #         #     fmap_[0] = fmap_[0][np.newaxis,:]
            #         #     fmap_ =
            #         # else:
            #         #     fmap_ = np.vstack(fmap_)
            #         fmap_ = xp.array(fmap_)
            #         src_fmap_dis.append(Variable(fmap_))
            else:
                if size != 0:
                    src_fmap_dis = []
                    for i in range(len(src_fmap)):
                        #src_fmap[i] = Variable(xp.vstack((src_fmap[i][0:batchsize - size], func_bGPU(e_buf_src[i]))))
                        src_fmap_dis.append(F.vstack((F.copy(src_fmap[i][0:batchsize - size],self.gpu_num), Variable(func_bGPU(e_buf_src[i])))))
                        src_fmap_dis[i].unchain_backward()
                else:
                    src_fmap_dis = []
                    for i in range(len(src_fmap)):
                        src_fmap_dis.append(F.copy(src_fmap[i],self.gpu_num))
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
                tgt_fmap_dis.append(F.copy(tgt_fmap[i][0:batchsize-size],self.gpu_num))
                tgt_fmap_dis[i].unchain_backward()
                if size > 0:
                    tgt_fmap_dis[i] = F.vstack([tgt_fmap_dis[i], Variable(func_bGPU(e_buf_tgt[i]))])

            y_target = self.dis(tgt_fmap_dis)
            # y_target_enc = self.dis(tgt_fmap)

            n_fmap_elements = y_target.shape[2]*y_target.shape[3]

            loss_dis_src = F.sum(F.softplus(-y_source)) / n_fmap_elements / batchsize
            loss_dis_tgt =  F.sum(F.softplus(y_target)) / n_fmap_elements / batchsize
            loss_dis = loss_dis_src + loss_dis_tgt

            loss_dis_src_sum += loss_dis_src.data
            loss_dis_tgt_sum += loss_dis_tgt.data
            loss_dis_sum += loss_dis.data

            self.dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()

            del loss_dis
            del src_fmap_dis
            del tgt_fmap_dis


        loss_dis_src_sum /= self.current_dis_step
        loss_dis_tgt_sum /= self.current_dis_step
        loss_dis_sum /= self.current_dis_step

        loss_t_enc_sum = 0
        loss_cls_sum = 0

        tempmodel = self.cls.copy()
        tempmodel.cleargrads()
        # self.cls.to_cpu()
        tempmodel.to_cpu()

        for i in range(self.current_tgt_step):
            #save fmap to buffer
            if i == 0 and self.bufmode == 1:
                src_fmap_tobuf = []
                tgt_fmap_tobuf = []
                for i in range(len(src_fmap)):
                    src_fmap_tobuf.append(chainer.cuda.to_cpu(src_fmap[i].data[:use_bufsize]))
                    tgt_fmap_tobuf.append(chainer.cuda.to_cpu(tgt_fmap[i].data[:use_bufsize]))
                self.buf.set_examples(src_fmap_tobuf, tgt_fmap_tobuf)

            # if i == 0 and self.batchmode == 1:
            #     y_target_enc = self.dis(tgt_fmap)
            #     mb_locs, mb_confs = self.cls.multibox(src_fmap)
            #     loc_loss, conf_loss = multibox_loss(
            #         mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
            #     cls_loss = loc_loss * self.alpha + conf_loss  # cls loss

            if i > 0 or self.bufmode == 0 or self.batchmode == 0:
                batch_source = self.get_iterator('main').next()
                batch_source_array = convert.concat_examples(batch_source, self.device)
                batch_target = self.get_iterator('target').next()
                src_fmap = self.t_enc(batch_source_array[0])  # src feature map
                tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))

            if i > 0 or self.batchmode == 0:
                y_target_enc = self.dis(tgt_fmap)
            loss_t_enc = F.sum(F.softplus(-y_target_enc)) / n_fmap_elements / batchsize

            #update cls(and t_enc) by cls_loss and loss_t_enc
            self.cls.cleargrads()
            loss_t_enc.backward()
            # if self.cls_train_mode == 1:
            #     cls_optimizer.update()

            if i > 0 or self.batchmode == 0:
                mb_locs, mb_confs = self.cls.multibox(src_fmap)
                loc_loss, conf_loss = multibox_loss(
                    mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
                cls_loss = loc_loss * self.alpha + conf_loss #cls loss

            # if self.cls_train_mode == 1:
            #     self.cls.cleargrads()

            cls_loss.backward()
            # cls_optimizer.update()
            tempmodel.addgrads(self.cls)

            # if i == 0 and self.bufmode == 0:
            #     src_fmap_tobuf = []
            #     tgt_fmap_tobuf = []
            #     for i in range(len(src_fmap)):
            #         src_fmap_tobuf.append(chainer.cuda.to_cpu(src_fmap[i].data[:use_bufsize]))
            #         tgt_fmap_tobuf.append(chainer.cuda.to_cpu(tgt_fmap[i].data[:use_bufsize]))
            #     self.buf.set_examples(src_fmap_tobuf, tgt_fmap_tobuf)

            for s_map, t_map  in zip(src_fmap, tgt_fmap):
                 s_map.unchain_backward()
                 t_map.unchain_backward()

            loss_t_enc_sum += loss_t_enc.data
            loss_cls_sum += cls_loss.data

            loss_t_enc.unchain_backward()
            del loss_t_enc
            cls_loss.unchain_backward()
            del cls_loss

            #coral loss
            self.cls.cleargrads()
            # arguments = {'s_imgs': batch_source_array[0], 'gt_mb_locs': batch_source_array[1],
            #              'gt_mb_labels': batch_source_array[2], 't_imgs': batch_target[:self.coral_batchsize]}
            # arguments['mode'] = 'CORAL'
            #
            # coral_loss = self.coral_loss_func(**arguments)
            batchsize_tgt = self.coral_batchsize
            # src_fmap = self.extractor(s_imgs)
            # tgt_fmap = self.extractor(t_imgs)
            src_examples = src_fmap[0][:batchsize_tgt]
            tgt_examples = tgt_fmap[0][:batchsize_tgt]
            n_data, c, w, h = src_examples.shape

            #coral loss calculation
            src_examples = F.im2col(src_examples, 3, 1, 1)
            src_examples = F.reshape(src_examples, (n_data, c, 3 * 3, w, h))
            src_examples = F.transpose(src_examples, axes=(0, 3, 4, 1, 2))
            src_examples = F.reshape(src_examples, (n_data * w * h, c * 3 * 3))
            tgt_examples = F.im2col(tgt_examples, 3, 1, 1)
            tgt_examples = F.reshape(tgt_examples, (n_data, c, 3 * 3, w, h))
            tgt_examples = F.transpose(tgt_examples, axes=(0, 3, 4, 1, 2))
            tgt_examples = F.reshape(tgt_examples, (n_data * w * h, c * 3 * 3))
            n_data = n_data * w * h
            norm_coef = 1 / (4 * (c * 3 * 3) ** 2)

            xp = self.cls.xp
            colvec_1 = xp.ones((1, n_data), dtype=np.float32)
            _s_tempmat = F.matmul(Variable(colvec_1), src_examples)
            _t_tempmat = F.matmul(Variable(colvec_1), tgt_examples)
            s_cov_mat = (F.matmul(F.transpose(src_examples), src_examples) - F.matmul(F.transpose(_s_tempmat),
                                                                                      _s_tempmat) / n_data) / n_data - 1 if n_data > 1 else 1
            t_cov_mat = (F.matmul(F.transpose(tgt_examples), tgt_examples) - F.matmul(F.transpose(_t_tempmat),
                                                                                      _t_tempmat) / n_data) / n_data - 1 if n_data > 1 else 1
            coral_loss = F.sum(F.squared_error(s_cov_mat, t_cov_mat)) * norm_coef * self.CORAL_weight

            loss_coral_sum += coral_loss.data
            coral_loss.backward()

            self.cls.addgrads(tempmodel)

            #last update
            cls_optimizer.update()

            coral_loss.unchain_backward()
            del coral_loss

        loss_t_enc_sum /= self.current_tgt_step
        loss_cls_sum /= self.current_tgt_step

        chainer.reporter.report({'loss_t_enc': loss_t_enc_sum})
        chainer.reporter.report({'loss_dis': loss_dis_sum})
        chainer.reporter.report({'loss_cls': loss_cls_sum})
        chainer.reporter.report({'loss_dis_src': loss_dis_src_sum})
        chainer.reporter.report({'loss_dis_tgt': loss_dis_tgt_sum})
        chainer.reporter.report({'loss_coral': loss_coral_sum})

class DA_updater1_buf_2_t_anno(chainer.training.StandardUpdater):
    def __init__(self, bufmode = 0,batchmode = 0, cls_train_mode = 0, init_disstep = 1, init_tgtstep = 1, tgt_steps_schedule = None, *args, **kwargs):
        self.dis, self.cls = kwargs.pop('models')
        self.buf = kwargs.pop('buffer')
        super(DA_updater1_buf_2_t_anno, self).__init__(*args, **kwargs)
        self.t_enc = self.cls.extractor
        self.alpha = 1
        self.k = 3
        self.bufmode = bufmode
        self.batchmode = batchmode
        self.cls_train_mode = cls_train_mode
        self.current_dis_step = init_disstep
        self.current_tgt_step = init_tgtstep
        self.tgt_steps_schedule = tgt_steps_schedule
        if self.tgt_steps_schedule != None:
            if isinstance(tgt_steps_schedule, list):
                self.tgt_steps_schedule.sort(key=lambda x:x[0])
            else:
                print("tgt step schedule must be specified by list object. The schedule is ignored.")
                self.tgt_steps_schedule = None

    def update_core(self):
        if isinstance(self.tgt_steps_schedule,list) and len(self.tgt_steps_schedule) > 0:
            while len(self.tgt_steps_schedule) > 0 and self.tgt_steps_schedule[0][0] < self.iteration:
                self.tgt_steps_schedule.pop(0)
            if len(self.tgt_steps_schedule) > 0:
                if self.tgt_steps_schedule[0][0] == self.iteration:
                    self.current_tgt_step = self.tgt_steps_schedule[0][1]
                    self.tgt_steps_schedule.pop(0)

        #t_enc_optimizer = self.get_optimizer('opt_t_enc')
        dis_optimizer = self.get_optimizer('opt_dis')
        cls_optimizer = self.get_optimizer('opt_cls')
        xp = self.dis.xp
        func_bGPU = (lambda x: chainer.cuda.to_gpu(x, device=self.device)) if self.device >= 0 else lambda x: x

        loss_dis_src_sum = 0
        loss_dis_tgt_sum = 0
        loss_dis_sum = 0

        try:
            src_fmaps_iter = self.get_iterator('src_fmaps')
            fix_src_encoder = True
        except KeyError:
            fix_src_encoder = False

        for z in range(self.current_dis_step):
            if not fix_src_encoder or (z == self.current_dis_step -1 and (self.bufmode ==1 or self.batchmode == 1)):
                batch_source = self.get_iterator('main').next()
                batchsize_source = len(batch_source)
                # if 'tgt_annotation' in self._iterators.keys():
                #     batch_source.extend(self.get_iterator('tgt_annotation').next())
                batch_source_array = convert.concat_examples(batch_source,self.device)
                src_fmap = self.t_enc(batch_source_array[0])  # src feature map
            batch_target = self.get_iterator('target').next()
            batchsize = len(batch_target)
            use_bufsize = int(batchsize/2)



            size = 0
            if batchsize >= 2:
                size, e_buf_src , e_buf_tgt = self.buf.get_examples(use_bufsize)
            if fix_src_encoder:
                src_fmap_dis = []
                batch_source_fixed = src_fmaps_iter.next()
                batchsize_fixed = len(batch_source_fixed)
                # batch_source_array_fixed = convert.concat_examples(batch_source_fixed, self.device)
                for i in range(len(batch_source_fixed[0])):
                    fmap_ = []
                    for j in range(batchsize_fixed):
                        fmap_.append(batch_source_fixed[j][i])
                    fmap_ = xp.array(fmap_)
                    src_fmap_dis.append(Variable(fmap_))
            else:
                if size != 0:
                    src_fmap_dis = []
                    for i in range(len(src_fmap)):
                        #src_fmap[i] = Variable(xp.vstack((src_fmap[i][0:batchsize - size], func_bGPU(e_buf_src[i]))))
                        src_fmap_dis.append(F.vstack((F.copy(src_fmap[i][0:batchsize - size],self.device), Variable(func_bGPU(e_buf_src[i])))))
                        src_fmap_dis[i].unchain_backward()
                else:
                    src_fmap_dis = []
                    for i in range(len(src_fmap)):
                        src_fmap_dis.append(F.copy(src_fmap[i],self.device))
                        src_fmap_dis[i].unchain_backward()

            y_source = self.dis(src_fmap_dis)

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

            loss_dis_src_sum += loss_dis_src.data
            loss_dis_tgt_sum += loss_dis_tgt.data
            loss_dis_sum += loss_dis.data

            self.dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()

        loss_dis_src_sum /= self.current_dis_step
        loss_dis_tgt_sum /= self.current_dis_step
        loss_dis_sum /= self.current_dis_step

        loss_t_enc_sum = 0
        loss_cls_sum = 0

        for i in range(self.current_tgt_step):
            #save fmap to buffer
            if i == 0 and self.bufmode == 1:
                src_fmap_tobuf = []
                tgt_fmap_tobuf = []
                for i in range(len(src_fmap)):
                    src_fmap_tobuf.append(chainer.cuda.to_cpu(src_fmap[i].data[:use_bufsize]))
                    tgt_fmap_tobuf.append(chainer.cuda.to_cpu(tgt_fmap[i].data[:use_bufsize]))
                self.buf.set_examples(src_fmap_tobuf, tgt_fmap_tobuf)

            if i == 0 and self.batchmode == 1:
                y_target_enc = self.dis(tgt_fmap)
                if 'tgt_annotation' in self._iterators.keys():
                    batch_tgt_anno = self.get_iterator('tgt_annotation').next()
                    t_anno_batchsize = len(batch_tgt_anno)
                    batch_tgt_anno_array = convert.concat_examples(batch_tgt_anno, self.device)
                    tgt_anno_fmap = self.t_enc(batch_tgt_anno_array[0])
                    src_fmap_ = []
                    for i in range(len(src_fmap)):
                        # src_fmap[i] = Variable(xp.vstack((src_fmap[i][0:batchsize - size], func_bGPU(e_buf_src[i]))))
                        src_fmap_.append(F.vstack((
                            F.copy(src_fmap[i][0:batchsize - t_anno_batchsize], self.device), tgt_anno_fmap[i])))
                    gt_locs = xp.vstack((batch_source_array[1][:batchsize-t_anno_batchsize], batch_tgt_anno_array[1]))
                    gt_labels = xp.vstack((batch_source_array[2][:batchsize - t_anno_batchsize], batch_tgt_anno_array[2]))
                else:
                    src_fmap_ = src_fmap
                    gt_locs = batch_source_array[1]
                    gt_labels = batch_source_array[2]
                mb_locs, mb_confs = self.cls.multibox(src_fmap_)
                loc_loss, conf_loss = multibox_loss(
                    mb_locs, mb_confs, gt_locs, gt_labels, self.k)
                cls_loss = loc_loss * self.alpha + conf_loss  # cls loss

            if i > 0 or self.bufmode == 0 or self.batchmode == 0:
                batch_source = self.get_iterator('main').next()
                batch_source_array = convert.concat_examples(batch_source, self.device)
                batch_target = self.get_iterator('target').next()
                src_fmap = self.t_enc(batch_source_array[0])  # src feature map
                tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))

            if i > 0 or self.batchmode == 0:
                y_target_enc = self.dis(tgt_fmap)
            loss_t_enc = F.sum(F.softplus(-y_target_enc)) / n_fmap_elements / batchsize

            #update cls(and t_enc) by cls_loss and loss_t_enc
            self.cls.cleargrads()
            loss_t_enc.backward()
            if self.cls_train_mode == 1:
                cls_optimizer.update()

            if i > 0 or self.batchmode == 0:
                # mb_locs, mb_confs = self.cls.multibox(src_fmap)
                # loc_loss, conf_loss = multibox_loss(
                #     mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
                # cls_loss = loc_loss * self.alpha + conf_loss #cls loss
                if 'tgt_annotation' in self._iterators.keys():
                    batch_tgt_anno = self.get_iterator('tgt_annotation').next()
                    t_anno_batchsize = len(batch_tgt_anno)
                    batch_tgt_anno_array = convert.concat_examples(batch_tgt_anno, self.device)
                    tgt_anno_fmap = self.t_enc(batch_tgt_anno_array[0])
                    src_fmap_ = []
                    for i in range(len(src_fmap)):
                        # src_fmap[i] = Variable(xp.vstack((src_fmap[i][0:batchsize - size], func_bGPU(e_buf_src[i]))))
                        src_fmap_.append(F.vstack((
                            F.copy(src_fmap[i][0:batchsize - t_anno_batchsize], self.device), tgt_anno_fmap[i])))
                    gt_locs = xp.vstack((batch_source_array[1][:batchsize-t_anno_batchsize], batch_tgt_anno_array[1]))
                    gt_labels = xp.vstack((batch_source_array[2][:batchsize - t_anno_batchsize], batch_tgt_anno_array[2]))
                else:
                    src_fmap_ = src_fmap
                    gt_locs = batch_source_array[1]
                    gt_labels = batch_source_array[2]
                mb_locs, mb_confs = self.cls.multibox(src_fmap_)
                loc_loss, conf_loss = multibox_loss(
                    mb_locs, mb_confs, gt_locs, gt_labels, self.k)
                cls_loss = loc_loss * self.alpha + conf_loss  # cls loss

            if self.cls_train_mode == 1:
                self.cls.cleargrads()

            cls_loss.backward()
            cls_optimizer.update()

            if i == 0 and self.bufmode == 0:
                src_fmap_tobuf = []
                tgt_fmap_tobuf = []
                for i in range(len(src_fmap)):
                    src_fmap_tobuf.append(chainer.cuda.to_cpu(src_fmap[i].data[:use_bufsize]))
                    tgt_fmap_tobuf.append(chainer.cuda.to_cpu(tgt_fmap[i].data[:use_bufsize]))
                self.buf.set_examples(src_fmap_tobuf, tgt_fmap_tobuf)

            for s_map, t_map  in zip(src_fmap, tgt_fmap):
                 s_map.unchain_backward()
                 t_map.unchain_backward()

            loss_t_enc_sum += loss_t_enc.data
            loss_cls_sum += cls_loss.data

        loss_t_enc_sum /= self.current_tgt_step
        loss_cls_sum /= self.current_tgt_step

        chainer.reporter.report({'loss_t_enc': loss_t_enc_sum})
        chainer.reporter.report({'loss_dis': loss_dis_sum})
        chainer.reporter.report({'loss_cls': loss_cls_sum})
        chainer.reporter.report({'loss_dis_src': loss_dis_src_sum})
        chainer.reporter.report({'loss_dis_tgt': loss_dis_tgt_sum})


class DA_updater_enc_only(chainer.training.StandardUpdater):
    def __init__(self,  *args, **kwargs):
        self.dis, self.cls = kwargs.pop('models')
        super(DA_updater_enc_only, self).__init__(*args, **kwargs)
        self.t_enc = self.cls.extractor
        self.alpha = 1
        self.k = 3

    def update_core(self):
        #t_enc_optimizer = self.get_optimizer('opt_t_enc')
        # dis_optimizer = self.get_optimizer('opt_dis')
        cls_optimizer = self.get_optimizer('opt_cls')
        xp = self.dis.xp

        batch_source = self.get_iterator('main').next()
        batch_source_array = convert.concat_examples(batch_source,self.device)
        src_fmap = self.t_enc(batch_source_array[0])  # src feature map
        batch_target = self.get_iterator('target').next()
        batchsize = len(batch_target)

        tgt_fmap = self.t_enc(Variable(xp.array(batch_target)))
        y_target_enc = self.dis(tgt_fmap)
        mb_locs, mb_confs = self.cls.multibox(src_fmap)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, batch_source_array[1], batch_source_array[2], self.k)
        cls_loss = loc_loss * self.alpha + conf_loss  # cls loss

        n_fmap_elements = y_target_enc.shape[2] * y_target_enc.shape[3]
        loss_t_enc = F.sum(F.softplus(-y_target_enc)) / n_fmap_elements / batchsize

        #update cls(and t_enc) by cls_loss and loss_t_enc
        self.cls.cleargrads()
        loss_t_enc.backward()
        cls_loss.backward()
        cls_optimizer.update()

        chainer.reporter.report({'loss_t_enc': loss_t_enc})
        chainer.reporter.report({'loss_cls': cls_loss})


