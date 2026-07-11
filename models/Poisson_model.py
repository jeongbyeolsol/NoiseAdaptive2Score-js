import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from skimage.measure import compare_psnr
import warnings
warnings.filterwarnings('ignore')
from util.util import calc_psnr
import numpy as np
import math
from .ema import ExponentialMovingAverage
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.optimize as optimize
from numpy.random import rand
import random
    
class PoissonModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')

        return parser
    def DCLoss(self,img, patch_size):
        """
        calculating dark channel of image, the image shape is of N*C*W*H
        """
        img = img.cuda()
        maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
        dc = maxpool(0-img[:, None, :, :, :])
    
        target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()) 
     
        loss = L1Loss(size_average=True)(-dc, target)
        return loss    
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['f','sigma']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['lr','score','recon']
        if self.isTrain:
            self.model_names = ['f']
        else:  # during test time, only load G
            self.model_names = ['f']
            self.visual_names = ['lr','score','recon']
        self.netf = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_f = torch.optim.Adam(self.netf.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_f)
        self.variance1 = (opt.sigma/255)**2
        self.batch = opt.batch_size  
        self.sigma_min = 0.03
        self.sigma_max = 0.1
        self.sigma_annealing = 2*11000
        self.target_model = opt.target_model
        self.acc= 0
        self.sigmas = np.exp(np.linspace(np.log(self.sigma_max), np.log(self.sigma_min),self.sigma_annealing))
        self.sigmas = torch.from_numpy(self.sigmas)
        self.ema = ExponentialMovingAverage(self.netf.parameters(), decay=0.999)
      
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB' # 데이터셋의 A 영역을 입력으로 받고, B 영역을 출력/정답 쪽으로 사용한다는 뜻 (현 코드에서 AtoB가 True이든 False이든 항상 데이터셋의 A만 가져옴 )
        self.hr = input['A' if AtoB else 'A']#.to(self.device,dtype = torch.float32)    # clean image
        self.lr = np.random.poisson(self.hr.numpy()/self.phi_s)*self.phi_s              # noisy image
        self.image_paths = input['A_paths' if AtoB else 'A_paths']        
        self.hr = self.hr.to(self.device,dtype = torch.float32)
        self.lr = torch.from_numpy(self.lr).to(self.device,dtype = torch.float32)
        
    def set_input_val(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.hr = input['A' if AtoB else 'A'].to(self.device,dtype = torch.float32)
        self.lr =input['B' if AtoB else 'B'].to(self.device,dtype = torch.float32)
        self.image_paths = input['A_paths' if AtoB else 'B_paths'] 
        
    def set_phi(self, iter):       
        self.phi_s = np.random.uniform(0.01, 0.05, size=1)
        """
        min_log = np.log([0.005])
        self.phi_now = 0.1
        phi_s = min_log + np.random.rand(1) * (np.log([self.phi_now]) - min_log)
        self.phi_s = np.exp(phi_s)
        """
        
    def set_sigma(self, iter):
        labels = torch.randint(0, len(self.sigmas), (self.lr.shape[0],))
        self.sigma = self.sigmas[labels].view(self.lr.shape[0], *([1] * len(self.lr.shape[1:]))).to(self.device,dtype = torch.float32)
        self.loss_sigma = self.sigma[0]
    
    def foward_estimation(self,noise_model):
        def estimate(noise_model):
            if noise_model == "Gaussian":
                self.recon = self.forward_search_gau()
            elif noise_model == "Poisson":
                self.recon = self.forward_search_poi()
            else:
                self.recon = self.forward_search_gamma()
            return self.recon
        return estimate(noise_model)
    def forward_search_gau(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if hasattr(self, "loaded_state"):
            self.ema.load_state_dict(self.loaded_state)

        self.ema.copy_to(self.netf.parameters())        
        self.noise = 1e-5 * torch.randn(self.lr.shape).to(self.device,dtype = torch.float32)
        self.score = self.netf(self.lr,0)[0]        
        self.score_2 = self.netf(self.lr+self.noise,0)[0]              
        self.noise_level = -self.noise/(self.score_2 - self.score)
        self.noise_level = torch.clamp(self.noise_level,0,1)
        self.noise_level = np.sqrt(torch.median(self.noise_level).cpu().detach().numpy())
        self.noise_level = self.noise_level*255
        self.recon = self.lr +(self.noise_level/255)**2 *(self.score)
        return self.recon
    
    
    
    def forward_search_gamma(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.ema.load_state_dict(self.loaded_state)
        self.ema.copy_to(self.netf.parameters())        
        self.noise = 1e-6 * torch.randn(self.lr.shape).to(self.device,dtype = torch.float32)
        self.score = self.netf(self.lr,0)[0]        
        self.score_2 = self.netf(self.lr+self.noise,0)[0]              
        a = (self.score_2 - self.score)
        b = 1/(self.lr+self.noise) - 1/(self.lr)
        self.noise_level = b/(a+b)
        self.noise_level = (torch.median(self.noise_level).cpu().detach().numpy())
        self.noise_level = np.around(self.noise_level,decimals= 2)
        nom = self.lr
        denom = (1-self.noise_level)- self.noise_level*self.lr*self.score
        self.recon = nom/denom
        return self.recon
    
    def noise_model_estimation(self,score):
        epsilon = 1e-5
        self.n = torch.randn(self.lr.shape).to(self.device,dtype = torch.float32)
        self.noise = epsilon * self.n
        y_e = self.lr+self.noise
        score_e = self.netf(y_e,0)[0]
        w = 2*(y_e*score_e - self.lr*score).cpu().detach().numpy() 
        a = torch.log(y_e/self.lr).cpu().detach().numpy()       
        b = (2*self.lr*score).cpu().detach().numpy()        
        ww = w/(b+2.2)
        idx = (ww <= 1e-6) & (ww >= -1e-6)
        w = w[idx]
        b = b[idx]
        w = np.nanmean(w)
        b = np.nanmean(b)
        first = a*(b-2)
        second = 4*a*(- 2*a*b + w)
        sqrt = (first)**2 - second
        sqrt = np.sqrt(sqrt)    
        p1 = (-first + sqrt)/(2*a)
        p2 = (-first - sqrt)/(2*a)
        p1 = np.nanmean(p1)
        p2 = np.nanmean(p2)
        p = max(p1,p2)
        P = max(p,0)
        return p
    
    def _load_ema_state_once(self):
        """체크포인트에서 읽은 EMA 상태가 있으면 최초 한 번만 적용한다."""
        if (
            hasattr(self, "loaded_state")
            and not getattr(self, "_ema_state_loaded", False)
        ):
            self.ema.load_state_dict(self.loaded_state)
            self._ema_state_loaded = True


    def forward(self):
        """
        학습 중 시각화를 위한 Poisson 복원.

        현재 학습 가중치를 보관하고 EMA 가중치로 복원한 뒤,
        반드시 원래 학습 가중치로 되돌린다.
        """
        self._load_ema_state_once()

        was_training = self.netf.training
        self.ema.store(self.netf.parameters())

        try:
            self.ema.copy_to(self.netf.parameters())
            self.netf.eval()

            with torch.no_grad():
                self.zeta = torch.as_tensor(
                    self.phi_s,
                    device=self.device,
                    dtype=torch.float32,
                )

                self.score = self.netf(self.lr, 0)[0]

                # exp overflow 방지
                exp_argument = torch.clamp(
                    self.zeta * self.score,
                    min=-20.0,
                    max=20.0,
                )

                self.recon = (
                    self.lr + 0.562 * self.zeta
                ) * torch.exp(exp_argument)

        finally:
            # 어떤 오류가 발생하더라도 학습 가중치를 복구한다.
            self.ema.restore(self.netf.parameters())
            self.netf.train(was_training)


    def forward_search_poi(self):
        """
        Poisson noise level을 추정하고 Tweedie 식으로 복원한다.

        EMA 교체는 forward_estimate()에서 담당하므로,
        이 함수에서는 EMA store/copy/restore를 하지 않는다.
        """
        with torch.no_grad():
            self.noise = 1e-5 * torch.randn_like(self.lr)

            self.score = self.netf(self.lr, 0)[0]
            self.score_2 = self.netf(
                self.lr + self.noise,
                0,
            )[0]

            score_diff = self.score_2 - self.score

            # score 차이가 0에 가까우면 division by zero가 발생한다.
            eps = 1e-12
            safe_sign = torch.where(
                score_diff >= 0,
                torch.ones_like(score_diff),
                -torch.ones_like(score_diff),
            )

            safe_score_diff = torch.where(
                score_diff.abs() < eps,
                safe_sign * eps,
                score_diff,
            )

            c = self.noise / safe_score_diff

            radicand = self.lr.square() - 2.0 * c

            # sqrt에 넣을 수 있는 유효한 원소만 선택한다.
            valid_radicand = (
                torch.isfinite(radicand)
                & torch.isfinite(safe_score_diff)
                & (radicand >= 0.0)
            )

            safe_radicand = torch.clamp(
                radicand,
                min=0.0,
            )

            noise_level_map = (
                -self.lr + torch.sqrt(safe_radicand)
            )

            valid_level = (
                valid_radicand
                & torch.isfinite(noise_level_map)
                & (noise_level_map > 0.0)
            )

            if not valid_level.any():
                raise FloatingPointError(
                    "유효한 Poisson noise-level 추정값이 없습니다. "
                    "score network의 학습이 아직 충분하지 않거나 "
                    "score difference가 지나치게 작을 수 있습니다."
                )

            # 유효한 양수 추정값의 median을 사용한다.
            estimated_level = torch.median(
                noise_level_map[valid_level]
            ).item()

            if not np.isfinite(estimated_level):
                raise FloatingPointError(
                    "추정된 Poisson noise level이 NaN 또는 Inf입니다."
                )

            # 복원 계산에는 반올림하지 않은 값을 사용한다.
            self.noise_level = max(
                float(estimated_level),
                1e-8,
            )

            # exponential overflow 방지
            exp_argument = torch.clamp(
                self.noise_level * self.score,
                min=-20.0,
                max=20.0,
            )

            self.recon = (
                self.lr + self.noise_level / 2.0
            ) * torch.exp(exp_argument)

            if not torch.isfinite(self.recon).all():
                raise FloatingPointError(
                    "Poisson 복원 영상에 NaN 또는 Inf가 발생했습니다."
                )

            return self.recon


    def forward_estimate(self):
        """
        과제의 noise model이 Poisson으로 고정되어 있으므로
        분포 판별 없이 Poisson noise level만 추정한다.

        EMA 가중치 교체는 이 함수 한 곳에서만 수행한다.
        """
        self._load_ema_state_once()

        was_training = self.netf.training
        self.ema.store(self.netf.parameters())

        try:
            self.ema.copy_to(self.netf.parameters())
            self.netf.eval()

            return self.forward_search_poi()

        finally:
            # validation 이후에는 반드시 원래 학습 가중치로 복구한다.
            self.ema.restore(self.netf.parameters())
            self.netf.train(was_training)


    def forward_psnr(self):
        """
        EMA 모델로 Poisson 복원을 수행하고 validation PSNR을 계산한다.

        forward_estimate()가 EMA 처리를 담당하므로,
        여기서는 EMA store/copy/restore를 다시 수행하지 않는다.
        """
        with torch.no_grad():
            reconstructed = self.forward_estimate()

            reconstructed = torch.clamp(
                reconstructed,
                min=0.0,
                max=1.0,
            )

            if not torch.isfinite(reconstructed).all():
                raise FloatingPointError(
                    "PSNR 계산 전 복원 영상에 NaN 또는 Inf가 있습니다."
                )

            recon_cpu = reconstructed.detach().cpu()
            hr_cpu = self.hr.detach().cpu()

            # 시각화 코드와의 호환성을 위해 저장한다.
            self.recon = recon_cpu

            psnr = calc_psnr(
                recon_cpu,
                hr_cpu,
            )

            return psnr
    
    def backward_f(self):
        """Calculate GAN and L1 loss for the generator"""            
        _,self.loss_f = self.netf(self.lr,self.sigma)     
        self.loss_f.backward()
        
    def optimize_parameters(self):        
        self.optimizer_f.zero_grad()        # set G's gradients to zero
        self.backward_f()                   # calculate graidents for G
        torch.nn.utils.clip_grad_norm_(self.netf.parameters(), 1)        
        self.optimizer_f.step()              # udpate G's weights              
        self.ema.update(self.netf.parameters())
        with torch.no_grad():
            self.forward()