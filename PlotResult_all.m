
im_root='I:\Dataset\Object Tracking\Benchmark_cvpr13\';
dt_root='I:\Dataset\Object Tracking\code_benchmark_yi\results\results_TRE_CVPR13\';

Method={'STC','KCF','Struck','DLT', 'Ours'};
% Video={'Jogging', 'Coke', 'Shaking', 'Car4', 'CarScale',  'Dog1', 'Trellis', 'Singer2','Skating1', 'David', 'Jumping', 'Tiger2', 'Fleetface', 'Bolt' };
% Data={'jogging-2', 'coke', 'Shaking','car4', 'carScale',  'dog1', 'trellis', 'singer2', 'skating1', 'david', 'jumping', 'tiger2', 'fleetface', 'bolt' };

Video={'Basketball', 'Fleetface', 'David', 'David3', 'Deer', 'Football', 'Jumping', 'Trellis'};

Color={[255,127,39]/255,'g','b','y','r'};

Line={'--','--','--','--','-'};

for jj=1:length(Video)

    close all;
    load([dt_root,lower(Video{jj}),'_CHAOscnew.mat']);
    tt=results{1}.res;
    startFrame=results{1}.startFrame;
    
    switch Video{jj}
        case 'Basketball'
            frame=[140 490 630];
        case 'Fleetface'
            frame=[400 640 680];
        case 'David'
            frame=[145 180 455];
        case 'David3'
            frame=[65 85 145];
        case 'Deer'
            frame=[20 25 35];
        case 'Football'
            frame=[140 215 300];
        case 'Jumping'
            frame=[66 100 175];
        case 'Trellis'
            frame=[285 400 420];
    end
            
    
    for tt=frame %1:5:size(tt,1)    % Lemming 200:5:280   %Jogging35:5:95
        figure(1), imshow(imread([im_root,Video{jj}, '\img\' num2str(tt+startFrame-1,'%04d.jpg')]));
%         set(gcf,'Position',[100 100 320 240]);
        axis image off,

        for ii=1:length(Method)
            
            if ii<length(Method)
                load([dt_root,lower(Video{jj}),'_',Method{ii} '.mat']);
                rt=results{1}.res;
                
                if ii==4
                    for k = 1:size(rt,1)
                        [rect c] = calcRectCenter(results{1}.tmplsize, rt(k,:));
                        rt2(k,:) = rect;
                    end
                    rt=rt2;
                end
                
            %                     center(i,:) = c;

            
%             if tt==380&&ii==4
%                 load([dt_root,Data{jj},'_TLD.mat']);
%                 mc=results{1}.res;
%                 hold on;
%                 rectangle('position',[mc(tt,1:2)-[4,8], rt(tt,3:4)],'LineWidth',6, 'edgecolor', Color{ii});
%                 continue;
%             end
            else
                load(['icip_result/' Video{jj},'_ICIP.mat']);
                rt=rect;
            end
            
            if sum(isnan(rt(tt,:)))<1
                hold on; rectangle('position',rt(tt,:),'LineWidth',4, 'edgecolor', Color{ii});
            else
                hold on; text(300,40, '×', 'color',Color{ii}, 'fontsize',60);
            end
        end
        
        hold off;
        
        text(20,30, num2str(tt+startFrame-1,'#%03d'),'color','c', 'fontsize',24, 'fontweight', 'bold');
    
%         if ~exist(['img_result/' Video{jj}], 'dir')
%           mkdir(['img_result/' Video{jj}]);
%         end
%   
%         f=getframe(gcf);
%         imwrite(frame2im(f), ['img_result/' Video{jj} '/' num2str(tt,'%04d.png')]);
        export_fig(['pdf_result/'  Video{jj} num2str(tt,'%04d.pdf')], '-transparent');
    end
end


            

% video='Jogging.2';
% 
% load(['.\MOSSE\' video '.mat']);
% rect1=rect;
% 
% load(['.\KCF\' video '.mat']);
% rect2=rect;
% 
% load(['.\STC\' video '.mat']);
% rect3=rect;
% 
% load(['.\CHAO\' video '.mat']);
% rect4=rect1;
% rect4(:,1:2)=positions(:,[2,1])-floor(rect4(:,[3,4])/2);
% 
% videoname='Jogging';
% % rt=['I:\Dataset\Object Tracking\Benchmark_cvpr13\' videoname '\img\'];
% 
% rt='I:\Chao Ma\Dropbox\Project\Tracking-Chao\Code ours\cvpr15_v1\data process\results\Jogging\';
% 
% for ii=35:5:95%size(rect,1)
%     
%     figure(1), imshow(imread([rt, num2str(ii,'%04d.jpg')]));
%     
%     axis image off,
%     
%     hold on; rectangle('position',rect1(ii,:),'LineWidth',3, 'edgecolor', 'g');
%     
%     hold on; rectangle('position',rect2(ii,:),'LineWidth',3, 'edgecolor', 'b');
%     
%     hold on; rectangle('position',rect3(ii,:),'LineWidth',3, 'edgecolor', 'y');
%     
%     hold on; rectangle('position',rect4(ii,:),'LineWidth',3, 'edgecolor', 'r');
%     
%     text(20,30, num2str(ii,'#%03d'),'color','c', 'fontsize',25);
%     
%     export_fig(['results/' videoname '/' num2str(ii,'%04d.jpg')]);
% end
