from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./logs/runs/train_20220407_165600')
x = range(100)
for i in x:
    #writer.add_scalar(tag='y=2x', scalar_value=i * 2, global_step=i)
    writer.add_scalar('y=2x', i * 2, i)
writer.close()
