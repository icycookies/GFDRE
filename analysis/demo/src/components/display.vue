<template>
    <el-container>
    <el-header> 文档级关系抽取演示</el-header>
    <el-main>
        标题/编号搜索：
        <el-input v-model="title_or_id"/>
        <el-button style="display:inline-block;margin-left: 15px;" v-on:click="show">展示</el-button>
        <h3> {{title}} </h3>
        <div style="text-align:left;margin-bottom:15px;" v-html="text"> </div>
        <div v-show="text.length > 0">
            请选择两个命名实体：
            <el-select v-model="q1">
                <el-option v-for="item in ents" :key="item.key" :label="item.value" :value="item.key"></el-option>
            </el-select>
            <el-select stype="margin-left:15px;" v-model="q2">
                <el-option v-for="item in ents" :key="item.key" :label="item.value" :value="item.key"></el-option>
            </el-select>
            <el-button v-on:click="displayrelation">抽取关系</el-button>
        </div>
        <div v-show="gt_rels.length > 0">
            <div> 模型预测：{{pred_rels}} </div>
            <div> 真实标签：{{gt_rels}} </div>
        </div>
        <div v-show="text.length > 0">
            <el-collapse>
                <el-collapse-item title="展示所有关系">
                    <el-table :data="all_rels">
                        <el-table-column prop="hname" label="头实体"></el-table-column>
                        <el-table-column prop="tname" label="尾实体"></el-table-column>
                        <el-table-column prop="r" label="关系"></el-table-column>
                        <el-table-column prop="is_gt" label="预测结果">
                            <template slot-scope="scope">
                                <el-tag
                                :type="scope.row.is_pred ? 'success' : 'danger'"
                                disable-transitions>{{scope.row.is_pred}}</el-tag>
                            </template>
                        </el-table-column>
                        <el-table-column prop="is_pred" label="真实标签">
                            <template slot-scope="scope">
                                <el-tag
                                :type="scope.row.is_gt ? 'success' : 'danger'"
                                disable-transitions>{{scope.row.is_gt}}</el-tag>
                            </template>
                        </el-table-column>
                    </el-table>
                    Precision: {{p.toFixed(3)}}
                    Recall: {{r.toFixed(3)}}
                    F1: {{f1.toFixed(3)}}
                </el-collapse-item>
            </el-collapse>
        </div>
    </el-main>
    </el-container>
</template>

<script>
import Docs from '../assets/result_eval.json'
export default {
    name: 'Display',
    props: {
    },
    data(){
        return {
            title_or_id: "",
            id2title: {},
            title: "",
            text: "",
            preds: {},
            gt: {},
            ents: [],
            q1: undefined,
            q2: undefined,
            all_gt_rels: [],
            all_pred_rels: [],
            gt_rels: [],
            pred_rels: [],
            all_rels: [],
            p: 0.0,
            r: 0.0,
            f1: 0.0,
        }
    },
    beforeMount(){
        let id = 0
        for (let doc in Docs){
            this.id2title[id] = doc
            id += 1
        }
    },
    methods: {
        show: function(){
            if (parseInt(this.title_or_id) != NaN){
                this.title = this.id2title[parseInt(this.title_or_id)]
            } else {
                this.title = this.title_or_id
            }
            let doc = Docs[this.title]
            let ent_ids = doc["tok2ent"]
            this.text = this.process_document(doc["doc"], ent_ids)
            this.ents = new Array(doc["ents"].length)
            for (let i = 0; i < doc["ents"].length; i++){
                this.ents[i] = {
                    "key": i,
                    "value": doc["ents"][i],
                }
            }
            this.all_gt_rels = doc["gt"]
            this.all_pred_rels = doc["pred"]
            this.all_rels = new Array(this.all_gt_rels.length)
            for (let i = 0; i < this.all_rels.length; i++){
                this.all_rels[i] = {
                    'h': this.all_gt_rels[i]['h'],
                    't': this.all_gt_rels[i]['t'],
                    'r': this.all_gt_rels[i]['r'],
                    "is_gt": true,
                    "is_pred": false,
                }
            }
            console.log(this.all_rels)
            let tp = 0
            for (let i = 0; i < this.all_pred_rels.length; i++){
                let rel1 = this.all_pred_rels[i]
                console.log(rel1)
                let flag = false
                for (let j = 0; j < this.all_rels.length; j++){
                    let rel2 = this.all_rels[j]
                    if (rel1['h'] == rel2['h'] && rel1['t'] == rel2['t'] && rel2['is_gt']){
                        this.all_rels[j]['is_pred'] = true
                        flag = true
                        tp += 1
                        break
                    }
                }
                if (!flag){
                    this.all_rels.splice(0, 0, {
                        'h': rel1['h'],
                        't': rel1['t'],
                        'r': rel1['r'],
                        'is_gt': false,
                        'is_pred': true,
                    })
                }
            }

            let cmp = function(a, b){
                if (a['h'] != b['h'])return a['h'] - b['h']
                    else return a['t'] - b['t']
            }
            console.log(this.all_rels)
            this.all_rels.sort(cmp)
            console.log(this.all_rels)
            for (let i = 0; i < this.all_rels.length; i++){
                this.all_rels[i]['hname'] = '[' + this.ents[this.all_rels[i]['h']]['key'] + '] ' + this.ents[this.all_rels[i]['h']]['value']
                this.all_rels[i]['tname'] = '[' + this.ents[this.all_rels[i]['t']]['key'] + '] ' + this.ents[this.all_rels[i]['t']]['value']
            }
            this.p = tp * 1.0 / this.all_pred_rels.length
            this.r = tp * 1.0 / this.all_gt_rels.length
            this.f1 = 2 * this.p * this.r / (this.p + this.r + 1e-20)
        },
        displayrelation: function(){
            this.gt_rels = []
            this.pred_rels = []
            console.log(this.all_gt_rels)
            for (let i = 0; i < this.all_gt_rels.length; i++){
                let rel = this.all_gt_rels[i]
                if (rel['h'] == this.q1 && rel['t'] == this.q2)this.gt_rels.splice(0, 0, rel['r'])
            }
            for (let i = 0; i < this.all_pred_rels.length; i++){
                let rel = this.all_pred_rels[i]
                if (rel['h'] == this.q1 && rel['t'] == this.q2)this.pred_rels.splice(0, 0, rel['r'])
            }
            if (this.gt_rels.length == 0)this.gt_rels.splice(0, 0, 'NA')
            if (this.pred_rels.length == 0)this.pred_rels.splice(0, 0, 'NA')
            this.gt_rels = this.gt_rels.join(";")
            this.pred_rels = this.pred_rels.join(";")
        },
        process_document: function(doc, ent_ids){
            let s = doc.split(" ")
            let result = ""
            console.log(s, ent_ids)
            for (let i = 0; i < s.length; i++){
                if (ent_ids[i] != -1 && (i == 0 || ent_ids[i - 1] == -1)){
                    result = result + "<b>"
                    result = result + '[' + String(ent_ids[i]) + ']'
                }
                result = result + s[i] + " "
                if  (ent_ids[i] != -1 && (i ==  s.length - 1 || ent_ids[i + 1] == -1)){
                    result = result + "</b>"
                }
            }
            return result
        }
    }
}
</script>

<style scoped>
    .el-container {
        width: 1000px;
        margin: 0 auto;
    }
    .el-header {
        background-color: #B3C0D1;
        color: #333;
        line-height: 60px;
    }
    .el-footer {
        background-color: #B3C0D1;
        color: #333;
        line-height: 60px;
    }
    .el-aside {
        color: #333;
    }
    .el-input {
        width: 400px;
    }
</style>