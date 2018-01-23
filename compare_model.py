import caffe
import numpy as np

net1 = caffe.Net('caffe_proto_1.prototxt', 'caffe_model_1.caffemodel', caffe.TEST)
net2 = caffe.Net('caffe_proto_2.prototxt', 'caffe_model_2.caffemodel', caffe.TEST)
final_result = True

for k, v in net1.params.items():
    print "first net:" + k
    if k in net2.params:
        print "        also exist in second net"
        w1 = net1.params[k][0].data[...]
        w2 = net2.params[k][0].data[...]
        if np.array_equal(w1,w2):
            print "        and has the same weight"
        else:
            print "        but has different weight"
            print "        w1"
            print w1
            print "        w2"
            print w2
            final_result = False
        if len(net1.params[k]) > 1:
            b1 = net1.params[k][1].data[...]
            b2 = net2.params[k][1].data[...]
            if np.array_equal(w1,w2):
                print "        and has the same bias"
            else:
                print "        but has different bias"
                final_result = False

    else:
        print "        not exist in second net"
        final_result = False
        break

for k, v in net2.params.items():
    print "second net:" + k
    if k in net1.params:
        print "        also exist in first net"
        w1 = net1.params[k][0].data[...]
        w2 = net2.params[k][0].data[...]
        if np.array_equal(w1,w2):
            print "        and has the same weight"
        else:
            print "        but has different weight"
            final_result = False
        if len(net1.params[k]) > 1:
            b1 = net1.params[k][1].data[...]
            b2 = net2.params[k][1].data[...]
            if np.array_equal(w1,w2):
                print "        and has the same bias"
            else:
                print "        but has different bias"
                final_result = False

    else:
        print "        not exist in first net"
        final_result = False
        break


if final_result == True:
    print "two model are the same"
else:
    print "two model are different"

