# 数据结构

## 单调队列/单调栈

单调栈维护的严格递减或递增或者非严格递减的的序列

用来找区间最大值/最小值，保持集合的高度有效性和秩序性。

单调队列deque要记录下标  用while

~~~c++
//二维单调队列   x轴缩点
for (int i = 1; i <= n; i++) {
	deque<int>q;      
	for (int j = 1; j <= m; j++) {
		if (!q.empty() && q.front() < j - k + 1) q.pop_front();  //超出队列集合范围
		while (!q.empty() && a[i][q.back()] < a[i][j]) q.pop_back();//踢出小于它的
		q.push_back(j);            
		if (j >= k) dp[i][j] = a[i][q.front()];
	}
}
//对第二维进行操作
for (int j = k; j <= m; j++) {
		deque<int>q;
		for (int i = 1; i <= n; i++) {
			if (!q.empty() && q.front() < i - k + 1) q.pop_front();
			while (!q.empty() && dp[q.back()][j] < dp[i][j]) q.pop_back();
			q.push_back(i);
			if (i >= k)   sum += dp[q.front()][j];
		}
	}
~~~

~~~c++
//单调队列加尺取   
int a[maxn],b,c,n,k;
int main(){
	cin>>n>>k;
	cin>>a[0]>>b>>c;
	int l=1,r;
	ll ans=0;
	deque<int>q1,q2;
	for(r=1;r<=n;r++){
		a[r]=(1ll*a[r-1]*b+c)%mod;
		while(!q1.empty()&&a[q1.back()]>=a[r]) q1.pop_back();
		q1.push_back(r);
		while(!q2.empty()&&a[q2.back()]<=a[r]) q2.pop_back();
		q2.push_back(r);
		while(a[q2.front()]-a[q1.front()]>k){
				ans+=n-r+1;
                 l++;
				while(q2.front()<l) q2.pop_front();
				while(q1.front()<l) q1.pop_front();
			}
		}
		cout<<ans<<'\n';
	}
~~~



## 哈希表

1.根据 + * % 这对元素造一个hash表（vector<int\>v[maxn]），当给出元素让我们匹配的时候，我们直接相同操作来hash查找

~~~c++
//sum1是元素加法   sum2是元素乘法  最后sum1+sum2就是他们的hash值
for (int i = 1; i <= n; i++) {
		int sum1 = 0, sum2 = 1;
		for (int j = 0; j <= 5; j++) {          //hash值操作
			cin >> a[i][j];
			sum1 += a[i][j];
			sum2 = 1ll * sum2 * a[i][j]%mod;
			sum1 %= mod;
			sum2 %= mod;
		}
		if (w1) continue;
		int k = sum1 + sum2;
		for (int p = 0;  p< mp[k].size(); p++) {    //进行hash查找并且匹配
			if (ck(mp[k][p], i)) {
				w1 = 1;
				break;
			}
		}
		mp[k].push_back(i);
}
~~~

2.字符串映射成整数，可求区间字符串的hash

base：131   每次区间值为h[r]-h[l-1]*p[r-l+1];

注意开ull    从1开始  s[i]-'a'+1

~~~c++
const int maxn = 1e6 + 5;
const int mod = 1e9+7;
ull h[maxn];
ull p[maxn];

char s[maxn];
void run() {
	cin>>s+1;
	int len=strlen(s+1);
	p[0]=1; //p^0=1
	for(int i=1;i<=len;i++){
		h[i]=h[i-1]*131+s[i]-'a'+1;  //从1-26选择   
		p[i]=p[i-1]*131;
	}
	int n;
	cin>>n;
	for(int i=1;i<=n;i++){
		int a1,b1,a2,b2;
		cin>>a1>>b1>>a2>>b2;

		if(h[b1]-h[a1-1]*p[b1-a1+1]==h[b2]-h[a2-1]*p[b2-a2+1]) cout<<"Yes"<<'\n' ;
		else cout<<"No"<<'\n';
	}
}

~~~



## 字符串



### next数组

next[i]表示“A中以i结尾的非前缀子串”与“A的前缀”能够匹配的最长长度

如果 i-next[i]能够整除i  那么从1到 i 有i/(i-next[i])个循环元 否则一定没有  充要条件

~~~c++
char a[maxn];
int kc = 0;
int nex[maxn];
void get_next() {
	nex[1] = 0;
	int len = strlen(a + 1);
	for (int i = 2, j = 0; i <= len; i++) {
		while (j && a[i] != a[j + 1]) j = nex[j];    //判断当前可以接在哪一段的
		if (a[i] == a[j + 1]) j++;
		nex[i] = j;
	}
}
~~~



## 字典树

难点在于查询

~~~c++
char s1[1000];
int tree[maxn][30];
int tot=1;
int del[maxn];
void insert(char s[]) {  //插入
	int len = strlen(s);
	int root = 1;
	for (int i = 0; i < len; i++) {
		int id = s[i] - 'a';
		if (!tree[root][id]) tree[root][id] = ++tot;
		root = tree[root][id];
	}
	del[root]++;
}
int find_(char str[])//查询操作，按具体要求改动
{
	int len = strlen(str);
	int root = 1,res=0;
	for (int i = 0; i < len; i++)
	{
		int id = str[i] - 'a';
		root = tree[root][id];
		if(root==0) return res;
		res+=del[root];
	}
	return res;
}
void run() {
 int n,m;
 cin>>n>>m;
 for(int i=1;i<=n;i++){
	cin>>s1;
	insert(s1);
 }
 for(int i=1;i<=m;i++){
 	cin>>s1;
 	cout<<find_(s1)<<'\n';
 }
}
~~~

~~~c++
//01串
int tree[maxn][2];
int tot = 1;
int del[maxn];
void insert(ll x) {
	int root = 1;
	for (int i = 30; i >= 0; i--) {
		int id;
		if (1 << i & x) id = 1;
		else id = 0;
		if (!tree[root][id]) tree[root][id] = ++tot;
		root = tree[root][id];
	}
}
ll find_(ll x)//查询操作，按具体要求改动
{
	int root = 1;
	ll res = 0;
	for (int i = 30; i >= 0; i--)
	{
		int id;
		if (1 << i & x) id = 1;
		else id = 0;
		if (tree[root][id ^ 1]) {
			root = tree[root][id ^ 1];
			res |= (1ll << i);
		}
		else root = tree[root][id];
	}
	return res;
}

~~~



## 二叉堆

### 链表    

双向链表和优先队列结合

凡是数据结构一定考虑边界，就是是否要放哨兵

~~~c++
const int maxn=1e5+10;
char s[maxn];
int l[maxn],r[maxn];
bool vis[maxn];
void remove(int x){
		l[r[x]]=l[x];
		r[l[x]]=r[x];
}
void run() {
	int n;
	cin>>n>>s+1;
	priority_queue<pii>q;
	for(int i=2;i<=n;i++){
		l[i]=i-1;           //初始化
		r[i]=i+1;
		if(!vis[i]&&s[i]==s[i-1]+1) q.push({s[i],i}),vis[i]=1;
		if(!vis[i-1]&&s[i]==s[i-1]-1) q.push({s[i-1],i-1}),vis[i-1]=1;
	}
	mset(vis,0);
	int ans=0;
	while(!q.empty()){
		pii it =q.top();
		int now=it.se;
		q.pop();
		if(vis[now]) continue;
		ans++;       
		if(l[now]==0||r[now]==n+1)continue;
		if(s[l[now]]==s[r[now]]-1) q.push({s[r[now]],r[now]});
		else if(s[l[now]]==s[r[now]]+1) q.push({s[l[now]],l[now]});
		remove(now);   //删点
		vis[now]=1;
	}
	cout<<ans<<'\n';
}

~~~



## 并查集

并查集适合维护具有传递性和连通性的关系

并查集是一片森林

1.并查集定位位置

~~~C++
for(int i=1;i<=n;i++){
	int w=N[i].w;
	int x=N[i].x;
	int y=N[i].y;
	int k=w-(query(y)-query(x-1));
	for(int j=1;j<=k;j++){
		int s=find(y);
        vis[s]=1;
		add(s,1);
		fa[s]=s-1;   //这个点指向s-1
		ans++;
	 }
 }
~~~

2.并查集边带权（信息具有传递性，核心关系就是d[],代表他和根的相对关系，用取模来表示他们的关系）

画图  找  x y  dx  dy  在不同关系下的状态

~~~c++
把关系看成一个图
d[x] 表示x到fa[x]的边权  每次合并时累加边权
 就是弄多一个数组查找和合并时记录下关系
int fa[30100],cnt[30100],d[300010];
int find(int x){
	if(x==fa[x]) return x;
	else{
		int root=find(fa[x]);
		d[x]+=d[fa[x]];
		fa[x]=root;
		return root;
  }
}
int main(){
	int n;
	cin>>n;
	for(int i=1;i<=30000;i++)fa[i]=i,cnt[i]=1;
	for(int i=1;i<=n;i++){
		string op;
		int x,y;
		cin>>op>>x>>y;
		int fx=find(x);
		int fy=find(y);
		if(op[0]=='M'){
		 if(fx!=fy){
			 d[fx]=cnt[fy];
			 fa[fx]=fy;
			 cnt[fy]+=cnt[fx];
		 }
	 }
	 else{
		if(fx!=fy) cout<<-1<<'\n';
		else cout<<abs(d[x]-d[y])-1<<'\n';
	  }
  }

}
~~~



3.并查集扩展域 (待补)

~~~c++
const int maxn=20010;
int n,m,fa[maxn<<1]; //扩展域  敌人的敌人是朋友
struct node{
	int x,y,w;
	bool operator<(const node o)const{
		return w>o.w;
	}
}N[100010];
int find(int x){return fa[x]==x?x:fa[x]=find(fa[x]);}
int main(){
	cin>>n>>m;
	for(int i=1;i<=m;i++){
		int x,y,w;
		cin>>x>>y>>w;
		N[i]={x,y,w};
	}
	sort(N+1,N+1+m);
	for(int i=1;i<=n<<1;i++) fa[i]=i;
	for(int i=1;i<=m;i++){
		int x=N[i].x;
		int y=N[i].y;
		int w=N[i].w;
		int fx=find(x),fy=find(y);
		if(fx==fy){
			cout<<w<<'\n';
			return 0;
		}
		//合并
		fa[fx]=find(y+n);   //指向补集
		fa[fy]=find(x+n);   //指向补集
	}
	cout<<0<<'\n';
}
~~~



## 树状数组

树状数组里面存的是前缀

~~~c++
普通版
int a[maxn];
int lowbit(int x){return x&-x;}
void add(int x,int w){ for(int i=x;i<maxn;i+=lowbit(i)) a[i]+=w;}
int query(int x){
	int res=0;
	for(int i=x;i;i-=lowbit(i)) res+=a[i];
	return res;
}
·························································································
差分版
void add(int x,int w){while(x<n) a[x]+=w,x+=x&-x;}
int query(int x){
    int res=0;
    while(x){
        res+=a[x];
        x-=x&-x;
    }
    return res;
}
int main(){
   int l,r,w;
   while(cin>>l>>r>>w){
   add(l,w),add(r+1,-w);for(int i=1;i<=10;i++) cout<<query(i)<<'\n';
   }
}
·························································································
进阶版
int a[100010],d1[100010],d2[100010];
int n=1000;
//单点修改
void add(int *arr,int x,int w){
    while(x<=n) arr[x]+=w,x+=x&-x;
}
//区间修改
void add2(int l,int r,int w){
    add(d1,l,w),add(d1,r+1,-w);
    add(d2,l,w*l),add(d2,r+1,-w*(r+1));
}
//单点查询
int query(int *arr,int x){
    int res=0;
    while(x) res+=arr[x],x-=x&-x;
    return res;
}
//区间查询
int query2(int l,int r){
    r++;
    return query(a,r)+r*query(d1,r)-query(d2,r)-
    (query(a,l-1)+l*query(d1,l-1)-query(d2,l-1));
}
—————————————————————————————————————————————————————————————————————————————————————————
//求前缀最大值/最小值   
void add(int x,int w){
    //w是该点的值
    while(x<=n)
      d[x]=max(w,d[x]),x+=x&-x;
}
int query(int x){
   int res=0;
   //res要初始化为最小
   while(x) res=max(res,d[x]),x-=x&-x;
   return res;
}
~~~

求第k大

树状数组+二分

~~~c++
const int maxn = 1e5 + 10;
int n;
int a[maxn], b[maxn], c[maxn];
int lowbit(int x) { return x & -x; }
void add(int x, int w) {
	for (int i = x; i < maxn; i += lowbit(i)) a[i] += w;
}
int query(int x) {
	int ans = 0;
	for (int i = x; i; i -= lowbit(i)) ans += a[i];
	return ans;
}
int find(int x) {
	int l = 1, r = maxn;
	while (l < r) {
		int mid = l + r >> 1;
		if (query(mid) >= x) r = mid;
		else l = mid + 1;

	}
	return l;
}
struct node {
	int x, id;
	bool operator<(const node& o)const {
		return x < o.x;
	}
}N[maxn];
int main() {
	std::ios::sync_with_stdio(false); std::cin.tie(0);
	int t;
	cin >> t;
	while (t--) {
		int ca;
		for (int i = 1; i < maxn; i++) a[i] = 0;
		cin >> ca >> n;
		cout << ca << ' ' << n - n / 2 << '\n';
		for (int i = 1; i <= n; i++) {
			cin >> N[i].x;
			N[i].id = i;
		}
		sort(N + 1, N + 1 + n);
		for (int i = 1; i <= n; i++) {
			c[N[i].id] = i;
			b[i] = N[i].x;
		}
		int d = 0;
		for (int i = 1; i <= n; i++) {
			add(c[i], 1);
			if (i & 1) {
				cout << b[find(i / 2 + 1)], d++;
				if (d % 10 == 0 && i != n) cout << '\n';
				else cout << ' ';
			}
		}
		cout << '\n';
	}
}
~~~



## 差分

~~~c++
//二维差分  前缀和和差分是互逆的  f(g(x))=g(f(x))=a;
开多一个数组记录差分
for (int i = 0; i < m; i++) {//m是修改操作次数
	int x1, y1, x2, y2, p;
	cin >> x1 >> y1 >> x2 >> y2 >> p;
	b[x1][y1] += p; b[x2 + 1][y2 + 1] += p;
	b[x2 + 1][y1] -= p; b[x1][y2 + 1] -= p;
}

前缀和操作
每个数+=前缀和操作
  for(int i=1;i<=n;i++){
	for(int j=1;j<=m;j++){
		s[i][j]=s[i-1][j]+s[i][j-1]-s[i-1][j-1];
		  s[i][j]+=b[i][j];
		 }
	 }
	ans=s[x2][y2]-s[x2-1][y1]-s[x1][y2-1]+s[x1][y1];
~~~



## 线段树

线段树一般用来解决可合并的区间问题
树上维护子树区间的问题也可以通过dfs序+线段树去维护
不要质疑线段树写法的正确性，首先检查做法是否正确
不要忘记build，dfs序问题build时注意下标
数组记得开4倍，push_down一定要想好顺序
对于每个问题，我们只要考虑清楚push_down和push_up的写法即可
当问题不好处理的时候，想象一下对于单次询问暴力的做法，再去想区间合并操作
对于求一个区间内权值大于y的最小下标x时，就先看左区间是不是能找到这样的值，找不到再找右区间
若找不到返回-1否则返回下标，用区间max<=y进行剪枝防止复杂度退化
TLE有可能是数据溢出的问题，检查函数传参的变量类型(是否需要long long)
TLE不要怀疑线段树的复杂度，要想想有没有优化或者哪里是不是死循环
多个update或者query检查是否调用错误

~~~c++
单个位置 in[x]   区间 (in[x],ou[x]) 建树(1,1,n) 查找更新都是1  搜索 dfs(k,-1) k是根     
void dfs(int rt,int fa){          //树上维护子区间
	in[rt]=++tim;  //时间戳
	dfn[tim]=rt;  //记录左边
	for(int i=0;i<G[rt].size();i++){
		 int to=G[rt][i];
		 if(to==fa)continue;
		 dfs(to,rt);
	}
	ou[rt]=tim;           //范围中的值
}
//线段树建树永远是从1开始建 无论dfn序根在哪
struct T
{
	int l,r,mid;
	int add,sum;
	int len(){return r-l+1;}
}tree[maxn<<2];
void push_up(int rt)
{
	tree[rt].val=max(tree[rt<<1].val,tree[rt<<1|1].val);   //往上区间合并
}

//无论什么标记  区间更新都要更新自己  然后下传懒标记
void push_down(int rt)             //往下传lazy标记 对儿子继续操作 负责更新儿子的操作
{
	if(tree[rt].add)
	{
		int tp=tree[rt].add;//在这里下传标记   
		//tree[rt<<1].add+=v   tree[rt<<1].sum+=.... 改变子区间的值和标记
		tree[rt].add=0;
	}
}
void build(int rt,int l,int r)
{
	tree[rt].add=0;
	tree[rt].l=l;
	tree[rt].r=r;
	if(l==r)
	{
		tree[rt].val=-1;          //单个节点之间赋值
		tree[rt].id=0;
		return ;
	}
	int mid=tree[rt].mid=l+r>>1;
	build(rt<<1,l,mid);
	build(rt<<1|1,mid+1,r);
	push_up(rt);
}

//区间更新+懒标记
void update(int rt,int l,int r)   //l ,r是更新的范围
{
	if(tree[rt].r<l||tree[rt].l>r) return ;     //r<l||l>r或者 val<val 剪枝
	if(tree[rt].l>=l&&tree[rt].r<=r)      
	{
	   //更新自己操作和sum val  //区间更新要用区间 
		return ;
	}
	push_down(rt);                                     //更新节点 查询时更新懒标记
	if(tree[rt].mid>=l) update(rt<<1,l,r);               //二分更新
	if(tree[rt].mid<r) update(rt<<1|1,l,r);
	push_up(rt);                                    //合并区间
}

//        单点更新
void update(int rt,int pos,int val)
{
	if(tree[rt].l==tree[rt].r)
	{
		tree[rt].GCD=val;
		return ;
	}
	if(pos<=tree[rt].mid) update(rt<<1,pos,val);
	else update(rt<<1|1,pos,val);
	push_up(rt);
}

//区间查询//也可以用来单点查询
int query(int rt,int l,int r)        //l ，r 是查询的范围
{
	if(tree[rt].r<l||tree[rt].l>r) return 0;
	if(tree[rt].l>=l&&tree[rt].r<=r)
		return// 返回本区间答案。
	push_down(rt);
	int ans=0;
	if(tree[rt].mid>=l) ans+=query(rt<<1,l,r);           //二分更新
	if(tree[rt].mid<r) ans+=query(rt<<1|1,l,r);
	push_up(rt);
	return ans;
}

//             单点查询  查询符合条件的尽可能单点
int query(int rt, int val, int l, int r)
{
	if (tree[rt].l >= l && tree[rt].r <= r)
	{
		if (tree[rt].val < val) return -1;
	}
	if (tree[rt].l == tree[rt].r) return tree[rt].id;
	int ans = -1;
	if (tree[rt].mid >= l)
	{
		ans = query(rt << 1, val, l, r);
		if (ans != -1) return ans;
	}
	if (tree[rt].mid < r)
	{
		ans = query(rt << 1 | 1, val, l, r);
		if (ans != -1) return ans;
	}
	return -1;
}

//有时候可以需要开多一个函数来找答案 
bool find(int rt, int l, int r, int x) {
	if (l == r) return 1;
	int a1 = x, a2 = x;
	if (tree[rt].mid >= l) a1 = _gcd(a1, query(rt << 1, l, r));           //二分更新
	if (tree[rt].mid < r) a2 = _gcd(a2, query(rt << 1 | 1, l, r));
	if (a1 == x && a2 == x) return 1;
	else if (a1 != x && a2 != x) return 0;
	else {
		if (a1 != x) return find(rt << 1, l, r, x);
		else return find(rt << 1 | 1, l, r, x);
	}
}

~~~



