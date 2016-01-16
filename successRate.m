
param='icip_result';

base_path='I:\Dataset\Object Tracking\Benchmark_cvpr13';

dirs = dir(base_path);
videos = {dirs.name};
videos(strcmp('.', videos) | strcmp('..', videos) | ...
    strcmp('anno', videos) | ~[dirs.isdir]) = [];

%the 'Jogging' sequence has 2 targets, create one entry for each.
%we could make this more general if multiple targets per video
%becomes a common occurence.
videos(strcmpi('Jogging', videos)) = [];
videos(end+1:end+2) = {'Jogging.1', 'Jogging.2'};


for ii=1:length(videos)
    
    video=videos{ii};
    
%     video_path = [base_path video '/'];
    
    load([param '/' video '_ICIP.mat'])
    
    [~, target_sz, ground_truth, ~] = load_video_info_mc(base_path, video);
    
%     rects = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
%     rects(:,3) = target_sz(2);
%     rects(:,4) = target_sz(1);

    ps=[];
    pc=[];
    for jj=1:size(rect,1)
        
        r1=rect(jj,:);
        r2=ground_truth(jj,:);
        
        ps(jj)=p_computePascalScoreRect(r1,r2);
        pc(jj)=p_computeCenterDistanceRect(r1, r2);
    end
    
    fs(ii)=sum(ps>=0.5)/length(ps);
    fc(ii)=sum(pc<=20)/length(pc);
    fca(ii)=mean(pc);
    
end

data=[videos' num2cell(fs') num2cell(fc') num2cell(fca')];

xlswrite('result_ICIP.xlsx', data, 1);