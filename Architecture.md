# 架构师视角回答架构设计方案
## 四步
+ 在回答系统复杂度来源的时候，要注意结合具体的业务场景和业务发展阶段来阐述。业务场景表明了业务的独特性，发展阶段表明了业务的成熟度，因为同一业务场景在不同阶段产生的矛盾也是不同的。

+ 在回答解决方案的时候，有价值的解决方案一定是建立在明确复杂度来源基础之上的。所以在设计架构的时候才分主要问题和次要问题，主要问题是必须要解决的点，次要问题可以根据实际情况进行取舍。

+ 在回答如何评估架构方案时，至少要从功能性和非功能性两个角度出发判断方案的合理性。对于很难决策的方案，要从更高的视角（比如技术负责人、业务负责人的角度）进行考量。

+ 在技术实现的细节上，要尽量讲出技术的实现原理，不要浮于表面的框架组合。

# 分布式数据存储

## 数据分片

+ Hash 
	+ 如何既保证数据分布均匀，又保证扩展性。
	+ 一致性哈希
	+ 仍无法避免单一热点问题。

+ Range分片
	+ 热门品类可以按照三级分类分片，非热门品类按照一级分类分片
	+ 要达到这种灵活性，需要分片元数据服务。

+ 最好是 Hash + Range

## 热点数据

+ 不是只加缓存就可以的，要区分场景。

+ 读操作：缓存和限流是任何时刻都需要做的，不分场景。 

+ 写操作：对于单一热点品类的数据存储写入问题，要做到有能力承载

## 分片元数据服务

###  设计
+ 给分片元数据做集群服务，并通过 ETCD 存储数据分片信息。

+ 每个数据存储实例节点定时向元数据服务集群同步心跳和分片信息。

+ 当调用端的请求过来时，元数据服务节点只需要做好高可用和缓存即可。

### 共识算法
+ ETCD 的共识算法是基于 Raft 协议实现的强一致性算法。
+ Paxos 又分为 Basic Paxos 和 Multi Paxos，然而因为它们的实现复杂，工业界很少直接采用 Paxos 算法，所以 ETCD 选择了 Raft 算法 
+ Raft 算法对于 Leader 领导者的选举是有限制的，只有最全的日志节点才可以当选
+ 共识算法的选择和数据副本数量的多少息息相关，如果副本少、参与共识的节点少，推荐采用广播方式，如 Paxos、Raft 等协议。如果副本多、参与共识的节点多，那就更适合采用 Gossip 这种最终一致性协议。



# 分布式系统事务一致性

## 2PC

+ 2PC 是分布式事务教父级协议，它是数据库领域解决分布式事务最典型的协议。它的处理过程分为准备和提交两个阶段，每个阶段都由协调者（Coordinator）和参与者（Participant）共同完成。

+ 准备阶段，事务管理器首先通知所有资源管理器开启事务，询问是否做好提交事务的准备。如资源管理器此时会将 undo 日志和 redo 日志计入事务日志中，并做出应答，当协调者接收到反馈 Yes 后，则准备阶段结束。

+ 提交阶段，当收到所有数据库实例的 Yes 后，事务管理器会发出提交指令。每个数据库接受指令进行本地操作，正式提交更新数据，然后向协调者返回 Ack 消息，事务结束。

+ 中断阶段，如果任何一个参与者向协调者反馈了 No 响应，例如用户 B 在数据库 D3 上面的余额在执行其他扣款操作，导致数据库 D3 的数据无法锁定，则只能向事务管理器返回失败。此时，协调者向所有参与者发出 Rollback 请求，参与者接收 Rollback 请求后，会利用其在准备阶段中记录的 undo 日志来进行回滚操作，并且在完成事务回滚之后向协调者发送 Ack 消息，完成事务回滚操作。

## 2PC问题

+ 资源死锁

+ 操作数据库行级锁

+ 数据不一致性，在提交阶段，出现了网络异常，只有部分数据库接收到请求，那么会导致未接收到请求的数据库无法提交事务。

+ 工程落地较复杂，除了金融等要求强一致性的场景，一般不用。

## 借助消息队列

+ MQ 自动应答机制导致的消息丢失？

+ 高并发场景下的消息积压导致消息丢失？

+ 双向消息确认的机制。

+ 举例：让订单系统把要发送的消息持久化到本地数据库里，然后将这条消息记录的状态设置为代发送，紧接着订单系统再投递消息到消息队列，优惠券系统消费成功后，也会向消息队列发送一个通知消息。当订单系统接收到这条通知消息后，再把本地持久化的这条消息的状态设置为完成。这样做后，即使最终 MQ 出现了消息丢失，也可以通过定时任务从订单系统的本地数据库中扫描出一段时间内未完成的消息，进行重新投递，最终保证订单系统和优惠券系统的最终事务一致性。

# 分布式系统中的锁

## 分布式锁经常遇到那些问题？

+ 可用问题：无论何时都要保证锁服务的可用性（这是系统正常执行锁操作的基础）。

+ 死锁问题：客户端一定可以获得锁，即使锁住某个资源的客户端在释放锁之前崩溃或者网络不可达（这是避免死锁的设计原则）。

+ 脑裂问题：集群同步时产生的数据不一致，导致新的进程有可能拿到锁，但之前的进程以为自己还有锁，那么就出现两个进程拿到了同一个锁的问题。

## 锁的四个设计原则

+ 互斥性：即在分布式系统环境下，对于某一共享资源，需要保证在同一时间只能一个线程或进程对该资源进行操作。

+ 高可用：也就是可靠性，锁服务不能有单点风险，要保证分布式锁系统是集群的，并且某一台机器锁不能提供服务了，其他机器仍然可以提供锁服务。

+ 锁释放：具备锁失效机制，防止死锁。即使出现进程在持有锁的期间崩溃或者解锁失败的情况，也能被动解锁，保证后续其他进程可以获得锁。

+ 可重入：一个节点获取了锁之后，还可以再次获取整个锁资源。

## 使用Mysql实现分布式锁
+ 可以。
+ 但并发天然劣势。

## 基于redis实现分布式锁(可用性问题)

+ 在加锁的过程中，实际上就是在给 Key 键设置一个值，为避免死锁，还要给 Key 键设置一个过期时间。

+ 解锁的过程就是将 lock_key 键删除，但不能乱删，要保证执行操作的客户端就是加锁的客户端。而这个时候， unique_value 的作用就体现出来，实现方式可以通过 lua 脚本判断 unique_value 是否为加锁客户端。

+ 选用 Lua 脚本是为了保证解锁操作的原子性。因为 Redis 在执行 Lua 脚本时，可以以原子性的方式执行，从而保证了锁释放操作的原子性。
 
## redis 分布式锁如何设置合理的超时时间？(解决死锁问题)

+ 可以基于续约的方式设置超时时间：先给锁设置一个超时时间，然后启动一个守护线程，让守护线程在一段时间后，重新设置这个锁的超时时间。实现方式就是：写一个守护线程，然后去判断锁的情况，当锁快失效的时候，再次进行续约加锁，当主线程执行完成后，销毁续约锁即可。

## Redis 如何解决集群情况下分布式锁的可靠性？（解决脑裂问题）

+ Redlock 算法: 我们假设目前有 N 个独立的 Redis 实例， 客户端先按顺序依次向 N 个 Redis 实例执行加锁操作。这里的加锁操作和在单实例上执行的加锁操作一样，但是需要注意的是，Redlock 算法设置了加锁的超时时间，为了避免因为某个 Redis 实例发生故障而一直等待的情况。当客户端完成了和所有 Redis 实例的加锁操作之后，如果有超过半数的 Redis 实例成功的获取到了锁，并且总耗时没有超过锁的有效时间，那么就是加锁成功。


# RPC

## 优化 RPC 的网络通信性能

+ 高并发下选择高性能的网络编程 I/O 模型，IO多路复用。

## 选型合适的 RPC 序列化方式

+ 选择合适的序列化方式，进而提升封包和解包的性能。

# 分布式系统中的MQ问题

## 如何确保消息不丢失

+ 在生产阶段，你需要捕获消息发送的错误，并重发消息。

+ 在存储阶段，你可以通过配置刷盘和复制相关的参数，让消息写入到多个副本的磁盘上，来确保消息不会因为某个 Broker 宕机或者磁盘损坏而丢失。

+ 在消费阶段，你需要在处理完全部消费业务逻辑之后，再发送消费确认。

## 如何避免重复消费

+ 一个幂等操作的特点是，其任意多次执行所产生的影响均与一次执行的影响相同。对于幂等的方法，不用担心重复执行会对系统造成任何改变。

+ 从对系统的影响结果来说：At least once + 幂等消费 = Exactly once。

+ 1.利用数据库的约束来防止重复更新数据：将事务消息用到的字段结合起来作为一个唯一约束Key， 在支持“INSERT IF NOT EXIST”语意的数据库中插入一条记录。

+ 2.可以为数据更新设置一次性的前置条件，来防止重复消息：给你的数据增加一个版本号属性，每次更数据前，比较当前数据的版本号是否和消息中的版本号一致，如果不一致就拒绝更新数据，更新数据的同时将版本号 +1

+ 3.用“记录并检查操作”的方式来保证幂等：在发送消息时，给每条消息指定一个全局唯一的 ID，消费时，先根据这个 ID 检查这条消息是否有被消费过，如果没有消费过，才更新数据，然后将消费状态置为已消费。实现难度较高，首先高可用的唯一ID系统就很复杂，其次，在“检查消费状态，然后更新数据并且设置消费状态”中，三个操作必须作为一组操作保证原子性，才能真正实现幂等，否则就会出现 Bug。

## 处理消息积压

+ 优化消息收发性能，预防消息积压的方法有两种，增加批量或者是增加并发，在发送端这两种方法都可以使用，在消费端需要注意的是，增加并发需要同步扩容分区数量，否则是起不到效果的。

+ 对于系统发生消息积压的情况，需要先解决积压，再分析原因，毕竟保证系统的可用性是首先要解决的问题。快速解决积压的方法就是通过水平扩容增加 Consumer 的实例数量。

## 如何利用事务消息实现一个分布式事务

+ 一个严格意义的事务实现，应该具有 4 个属性：原子性、一致性、隔离性、持久性。这四个属性通常称为 ACID 特性。
  
+ 没看懂。


# MySQL

## 读多写少

### 读写分离，一主多从，在写数据时只写主库，在读数据时只读从库。

### 主从复制过程
 
+ 写入 Binlog：主库写 binlog 日志，提交事务，并更新本地存储数据。

+ 同步 Binlog：把 binlog 复制到所有从库上，每个从库把 binlog 写到暂存日志中。

+ 回放 Binlog：回放 binlog，并更新存储数据。

### 主从复制模式

+ 同步复制：事务线程要等待所有从库的复制成功响应。

+ 异步复制：事务线程完全不等待从库的复制成功响应。

+ 半同步复制：MySQL 5.7 版本之后增加的一种复制方式，介于两者之间，事务线程不用等待所有的从库复制成功响应，只要一部分复制成功响应回来就行，比如一主二从的集群，只要数据成功复制到任意一个从库上，主库的事务线程就可以返回给客户端。

### 从架构上解决主从复制延迟
+ 使用数据冗余
+ 使用缓存解决
+ 直接查询主库

### 复制状态机
+ 如果客户端将要执行的命令发送给集群中的一台服务器，那么这台服务器就会以日志的方式记录这条命令，然后将命令发送给集群内其他的服务，并记录在其他服务器的日志文件中，注意，只要保证各个服务器上的日志是相同的，并且各服务器都能以相同的顺序执行相同的命令的话，那么集群中的每个节点的执行结果也都会是一样的。
+ 这种数据共识的机制就叫复制状态机，目的是通过日志复制和回放的方式来实现集群中所有节点内的状态一致性。
+ MySQL，Raft, Redis Cluster

## 写多读少
### 分库分表

### 如何确定分库还是分表？
+ 当数据量过大造成事务执行缓慢时，就要考虑分表。
+ 为了应对高并发，一个数据库实例撑不住，即单库的性能无法满足高并发的要求，考虑分库。

### 如何选择分片策略？
+ 垂直拆分是根据数据的业务相关性进行拆分，把不同的业务数据进行隔离，让系统和数据更为“纯粹”，关注业务的扩展。
+ 水平拆分指的是把单一库表数据按照规则拆分到多个数据库和多个数据表中，关注数据的扩展。
+ Range拆分
	+ 对热点数据做垂直扩展。
	+ 分片元数据服务。
+ 垂直水平拆分： 是综合垂直和水平拆分方式的一种混合方式，垂直拆分把不同类型的数据存储到不同库中，再结合水平拆分，使单表数据量保持在合理范围内，提升性能

# Redis

## 高性能
 
+ Redis 的大部分操作都在内存中完成，并且采用了高效的数据结构，比如哈希表和跳表。

+ 因为是单线程模型避免了多线程之间的竞争，省去了多线程切换带来的时间和性能上的开销，而且也不会导致死锁问题。
	+ 虽然 Redis 一直是单线程模型，但是在 Redis 6.0 版本之后，也采用了多个 I/O 线程来处理网络请求，这是因为随着网络硬件的性能提升，Redis 的性能瓶颈有时会出现在网络 I/O 的处理上，所以为了提高网络请求处理的并行度，Redis 6.0 对于网络请求采用多线程来处理。但是对于读写命令，Redis 仍然使用单线程来处理。

+ 最后，也是最重要的一点， Redis 采用了 I/O 多路复用机制。

## 高可用之持久化

+ AOF 日志（Append Only File，文件追加方式）：记录所有的操作命令，并以文本的形式追加到文件中。

+ RDB 快照（Redis DataBase）：将某一个时刻的内存数据，以二进制的方式写入磁盘。

+ 混合持久化方式：Redis 4.0 新增了混合持久化的方式，集成了 RDB 和 AOF 的优点。



# 缓存策略

## 缓存穿透问题

+ 查询缓存中不存在的数据时，每次都要查询数据库。

+ 给所有指定的 key 预先设定一个默认值，比如空字符串“Null”，当返回这个空字符串“Null”时，我们可以认为这是一个不存在的 key，在业务代码中，就可以判断是否取消查询数据库的操作，或者等待一段时间再请求这个 key。如果此时取到的值不再是“Null”，我们就可以认为缓存中对应的 key 有值了，这就避免出现请求访问到数据库的情况，从而把大量的类似请求挡在了缓存之中。

## 缓存并发问题

+ 假设在缓存失效的同时，出现多个客户端并发请求获取同一个 key 的情况，

+ SETNX 枷锁， 保证在同一时间只能有一个请求来查询数据库并更新缓存系统，其他请求只能等待重新发起查询，从而解决缓存并发的问题。

## 缓存雪崩

+ 缓存集体同时失效，如果此时请求并发很高，就会导致大面积的请求打到数据库，造成数据库压力瞬间增大，出现缓存雪崩的现象。

+ 将缓存失效时间随机打散

+ 设置缓存不过期。
## 如何设计一个缓存策略，可以动态缓存热点数据呢？

### LRU + 随机选取

+ 思路：就是通过判断数据最新访问时间来做排名，并过滤掉不常访问的数据，只留下经常访问的数据，具体细节如下。
	+ 先通过缓存系统做一个排序队列（比如存放 1000 个商品），系统会根据商品的访问时间，更新队列信息，越是最近访问的商品排名越靠前。
	+ 同时系统会定期过滤掉队列中排名最后的 200 个商品，然后再从数据库中随机读取出 200 个商品加入队列中。
	+ 这样当请求每次到达的时候，会先从队列中获取商品 ID，如果命中，就根据 ID 再从另一个缓存数据结构中读取实际的商品信息，并返回。
	+ 在 Redis 中可以用 zadd 方法和 zrange 方法来完成排序队列和获取 200 个商品的操作。

## 


# 如何向面试官证明你做的系统是高可用的

### 服务等级协议（Service-Level Agreement，SLA），LA 等于 4 个 9，也就是可用时长达到了 99.99% ，不可用时长则为是0.01%，

## 如何监控系统高可用？

### 基础设施监控
+ 系统要素指标：主要有 CPU、内存，和磁盘。
+ 网络要素指标：主要有带宽、网络 I/O、CDN、DNS、安全策略、和负载策略。
 
### 系统应用监控
+ 流量、耗时、错误、心跳、客户端数、连接数

### 存储服务监控
+ 除了基础指标监控之外，还有一些比如集群节点、分片信息、存储数据信息等相关特有存储指标的监控。


## 如何保证系统高可用？

### 熔断

+ 服务熔断其实是一个有限状态机，实现的关键是三种状态之间的转换过程。

+ 在这个状态机中存在关闭、半打开和打开三种状态。

	+ “关闭”转换“打开”：当服务调用失败的次数累积到一定的阈值时，服务熔断状态，将从关闭态切换到打开态。

	+ “打开”转换“半打开”：当熔断处于打开状态时，我们会启动一个超时计时器，当计时器超时后，状态切换到半打开态。

	+ “半打开”转换“关闭”：在熔断处于半打开状态时，请求可以达到后端服务，如果累计一定的成功次数后，状态切换到关闭态。

### 降级
+ 服务降级
	+ 读操作降级： 做数据兜底服务。
	+ 写操作降级： 同样的，将之前直接同步调用写数据库的操作，降级为先写缓存，然后再异步写入数据库。
	+ 读操作降级的设计原则，就是取舍非核心服务。
	+ 写操作降级的设计原则，就是取舍系统一致性，实现方式是把强一致性转换成最终一致性
+ 功能降级
	+ 在做产品功能上的取舍，既然在做服务降级时，已经取舍掉了非核心服务，那么同样的产品功能层面也要相应的进行简化。在实现方式上，可以通过降级开关控制功能的可用或不可用。

### 其余方式
+ 限流
+ 冗余
+ 负载均衡
+ 故障隔离

# 如何向面试官证明你做的系统是高性能的？

### 业务场景

+ 如果不考虑实际业务需求，这样的回答没有任何意义，因为高性能与业务是强相关的：

	+ 如果一台网络游戏服务器，可以支撑 2 百名玩家同时在线开黑，可能就算高性能；

	+ 如果一台网络直播服务器，可以支撑 2 千名用户同时在线观看，可能就算高性能；

	+ 如果一台电商平台服务器，可以支撑 2 万名用户同时在线下单，可能就算高性能；

### 指标
+ 吞吐量
+ RT
+ TP999

