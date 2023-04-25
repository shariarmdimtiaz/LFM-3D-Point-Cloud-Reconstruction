function [ points ] = depthimg2point( depthmap, pram )

[x,y]=find(depthmap>pram);
points=[x,y,depthmap(depthmap>pram)];

end

