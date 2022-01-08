from fastai.vision.all import *
from tqdm.notebook import tqdm

emb_size = 5
output_classes = 10

classifier = nn.Linear(emb_size, output_classes, bias=False)
W = classifier.weight.T # nn.Linear keeps it output_classes x emb_size but it's easier to think about it the other way round

counter = 0 

class ArcFaceClassifier(nn.Module):
    def __init__(self, emb_size, output_classes):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(emb_size, output_classes))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x):
        # Step 1:
        x_norm = F.normalize(x)
        W_norm = F.normalize(self.W, dim=0)
        # Step 2:
        return x_norm @ W_norm
   
    
def arcface_loss(cosine, targ, m=.4):
    # this prevents nan when a value slightly crosses 1.0 due to numerical error
    cosine = cosine.clip(-1+1e-7, 1-1e-7) 
    # Step 3:
    arcosine = cosine.arccos()
    # Step 4:
    arcosine += F.one_hot(targ, num_classes = output_classes) * m
    # Step 5:
    cosine2 = arcosine.cos()
    # Step 6:
    return F.cross_entropy(cosine2, targ)


def arcface_loss2(cosine, targ, m2 = 0.4, m1 = 1, m3 = 0.2):
	# this prevents nan when a value slightly crosses 1.0 due to numerical error
	cosine = cosine.clip(-1+1e-7, 1-1e-7)
	arcosine = cosine.arccos()
	arcosine += F.one_hot(targ, num_classes = output_classes) * m2
	cosine2 = arcosine.cos()
	cosine2 -= F.one_hot(targ, num_classes = output_classes) * m3

	return F.cross_entropy(cosine2, targ)


class SimpleConv(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        ch_in=[3,6,12,24,48,96]
        convs = [ConvLayer(c, c*2, stride=2) for c in ch_in]
        convs += [AdaptiveAvgPool(), Flatten(), nn.Linear(192, emb_size)]
        self.convs = nn.Sequential(*convs)
        self.classifier = classifier
        
    def get_embs(self, x):
        return self.convs(x)
    
    def forward(self, x):
        x = self.get_embs(x)
        x = self.classifier(x)
        return x

# helper method to extract all embedings from a data loader
def get_embs(model, dl):
    embs = []
    ys = []
    for bx,by in tqdm(dl):
        with torch.no_grad():
            embs.append(model.get_embs(bx))
            ys.append(by)
    embs = torch.cat(embs)
    embs = embs / embs.norm(p=2,dim=1)[:,None]
    ys = torch.cat(ys)
    return embs,ys

# helper to plot embeddings in 3D
def plot_embs(embs, ys, ax, title):
    #ax.axis('off')
    ax.set_title(title)
    for k in range(10):
        e = embs[ys==k].cpu()
        ax.scatter(e[:,0], e[:,1], e[:,2], s=4, alpha=.2)   

if __name__ == '__main__':
    path = 'cifar10'
    dls =ImageDataLoaders.from_folder(path, train='train', valid='test', num_workers=4)

    learn = Learner(dls, SimpleConv(ArcFaceClassifier(5,10)), metrics=accuracy, loss_func = arcface_loss)
    learn.fit_one_cycle(5, 5e-3)
    embs_arcface, ys_arcface  = get_embs(learn.model.eval(), dls.valid)

    learn = Learner(dls, SimpleConv(ArcFaceClassifier(5,10)), metrics=accuracy, loss_func = arcface_loss2)
    learn.fit_one_cycle(5, 5e-3)
    embs_arcface2, ys_arcface2  = get_embs(learn.model.eval(), dls.valid)

    learn = Learner(dls, SimpleConv(ArcFaceClassifier(5,10)), metrics=accuracy)
    learn.fit_one_cycle(5, 5e-3)
    embs_softmax, ys_softmax  = get_embs(learn.model.eval(), dls.valid)

    _,(ax1,ax2,ax3)=plt.subplots(1,3, figsize=(20,10), subplot_kw={'projection':'3d'})
    plot_embs(embs_softmax, ys_softmax, ax1, title = 'Classification results with Softmax')
    plot_embs(embs_arcface, ys_arcface, ax2, title = 'Classification results with ArcFace')
    plot_embs(embs_arcface2, ys_arcface2, ax3, title = 'Classification results with ArcFace2')
    plt.savefig('main_figure.png')


