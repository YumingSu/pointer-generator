from tensorboard.backend.event_processing import event_accumulator

#加载日志数据
ea=event_accumulator.EventAccumulator(r'D:\DeepLearingReporsity\pointer-generator\logs\cnndaily_adagrad\train_20220406_221602\events.out.tfevents.1649254563.SYMM.25488.0')
ea.Reload()
print(ea.scalars.Keys())

val_acc=ea.scalars.Items('val_acc')
print(len(val_acc))
print([(i.step,i.value) for i in val_acc])

import matplotlib.pyplot as plt
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(111)
val_acc=ea.scalars.Items('val_loss')
ax1.plot([i.step for i in val_acc],[i.value for i in val_acc],label='val_loss')
ax1.set_xlim(0)
acc=ea.scalars.Items('loss')
ax1.plot([i.step for i in acc],[i.value for i in acc],label='loss')
ax1.set_xlabel("step")
ax1.set_ylabel("")

plt.legend(loc='lower right')
plt.show()