// Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
function drawCombinedGraph(data) {
    $(".combined-graph").show();
    // drawCombinedGraph.chart is a static variable that holds the graph state;
    // it is initialized on first call to drawCombinedGraph()
    if (typeof drawCombinedGraph.chart == 'undefined') {
        // create instance of C3 chart
        drawCombinedGraph.chart = c3.generate($.extend({
            bindto: '#combined-graph',
            axis: {
                x: {
                    label: {
                        text: 'Epoch',
                        position: 'outer-center',
                    },
                    tick: {
                        // 3 sig-digs
                        format: function(x) { return Math.round(x*1000)/1000; },
                        fit: false,
                    },
                    min: 0,
                    padding: {left: 0},
                },
                y: {
                    label: {
                        text: 'Loss',
                        position: 'outer-middle',
                    },
                    min: 0,
                    padding: {bottom: 0},
                },
                y2: {
                    show: true,
                    label: {
                        text: 'Accuracy (%)',
                        position: 'outer-middle',
                    },
                    min: 0,
                    max: 100,
                    padding: {top: 0, bottom: 0},
                },
            },
            grid: {x: {show: true} },
            legend: {position: 'bottom'},
        },
        {
            data: data,
            transition: {
                duration: 0,
            },
            subchart: {
                show: true,
            },
            zoom: {
                rescale: true,
            },
        }
        ));
    }
    else
    {
        // just update data
        drawCombinedGraph.chart.load(data);
        drawCombinedGraph.chart.data.names(data.names);
    }
}
function drawLRGraph(data) {
    $(".lr-graph").show();
    c3.generate($.extend({
        bindto: '#lr-graph',
        size: {height: 300},
        axis: {
            x: {
                label: {
                    text: 'Epoch',
                    position: 'outer-center',
                },
                tick: {
                    // 3 sig-digs
                    format: function(x) { return Math.round(x*1000)/1000; },
                    fit: false,
                },
                min: 0,
                padding: {left: 0},
            },
            y: {
                label: {
                    text: 'Learning Rate',
                    position: 'outer-middle',
                },
                min: 0,
                padding: {bottom: 0},
            },
        },
        grid: {x: {show: true} },
        legend: {show: false},
    },
    {data: data}
    ));
}


function drawConfusionMatrix(cm_data) {
    $(".confusion-matrix").show();
    console.log('model-graphs.js : drawConfusionMatrix(..)');

    //@TODO: initialize nicely. we now remove the chart.
    d3.select("#confusion-matrix").html("");

    var labels = cm_data.labels;
    var n_labels = labels.length;
    var n_epoch = cm_data.cm.length;

    //Get longest string
    var longest_label_size = cm_data.labels.sort(function (a, b) { return b.length - a.length; })[0].length;
   // console.log(longest_label_size);
    var dynamic_margin = 20+longest_label_size*4;
    var margin = {top: dynamic_margin, right: 0, bottom: 0, left: dynamic_margin};

    var min_wh = Math.min($("#confusion-matrix").width(),$("#confusion-matrix").height());
    var width = min_wh- margin.left - margin.right;
    var height = min_wh-margin.bottom-margin.top;



    var svg = d3.select("#confusion-matrix").append("svg")
        .attr("width", min_wh + margin.left*2 + margin.right)
        .attr("height", min_wh+ margin.top + margin.bottom )
        .attr("font-size", "10px" )
        .style("margin-left", margin.left + "px")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var x = d3.scale.ordinal().rangeBands([0, width]),
        color = d3.scale.linear()
        .domain([0,20,40,60,80,100])
        .range(["#f0f9e8","#ccebc5","#a8ddb5","#7bccc4","#43a2ca","#0868ac"]);



    var refreshGraph = function(cm_data_ep, epoch) {
      console.log('refreshGraph(..,epoch=' + epoch + ")");
      d3.select("#cm_epoch_label")
        .text("Epoch "+epoch);
      svg.selectAll(".line").remove();
      svg.selectAll(".cm_column").remove();
      svg.selectAll(".cm_row").remove();
      svg.selectAll(".cm_row_acc").remove();

      var matrix = [];
      var nodes = [];
      var orders = [];
     
      function reloadData(epoch) {
          matrix = [];
          nodes = [];
          orders = [];
          for (i = 0; i < labels.length; i++) {
            nodes[i]=[];
            nodes[i].index = i;
            nodes[i].count = 0;
            nodes[i].name = labels[i];
            matrix[i] = d3.range(n_labels).map(function(j) { return {x: j, y: i, z: 0}; });
          }

          for (xi = 0; xi < cm_data_ep.cm[epoch].length; xi++) {
            for (y = 0; y < cm_data_ep.cm[epoch][xi].length; y++) {
              matrix[xi][y].z=cm_data_ep.cm[epoch][xi][y];
              nodes[xi].count += cm_data_ep.cm[epoch][xi][y];
              //nodes[y].count += cm_data_ep.matrix[xi][y]; // is this one needed?
            }
            nodes[xi].acc = (100*matrix[xi][xi].z/nodes[xi].count).toFixed(1) + '%';
          }

          // Precompute the orders.
          orders = {
            name: d3.range(n_labels).sort(function(a, b) { return d3.ascending(nodes[a].name, nodes[b].name); }),
            index: d3.range(n_labels).sort(function(a, b) { return b - a; }),
            count: d3.range(n_labels).sort(function(a, b) { return nodes[b].count - nodes[a].count; })
          };

          // The default sort order.
          x.domain(orders.name);
      }

      reloadData(epoch);

      

      svg.append("rect")
          .attr("class", "background")
          .attr("width", width)
          .attr("height", height);

      var row = svg.selectAll(".cm_row")
          .data(matrix)
          .enter().append("g")
          .attr("class", "cm_row")
          .attr("transform", function(d, i) { return "translate(0," + x(i) + ")"; })
          .each(row);

      row.append("line")
          .attr("x2", width);

      row.append("text")
          .attr("x", -6)
          .attr("y", x.rangeBand() / 2)
          .attr("dy", ".32em")
          .attr("text-anchor", "end")
          .text(function(d, i) { return nodes[i].name; });

      var row_acc = svg.selectAll(".cm_row_acc")
          .data(matrix)
          .enter().append("g")
          .attr("class", "cm_row_acc")
          .attr("transform", function(d, i) { return "translate(0," + x(i) + ")"; });

      row_acc.append("text")
          .attr("x", width+6)
          .attr("y", x.rangeBand() / 2)
          .attr("dy", ".32em")
          .attr("text-anchor", "start")
          .text(function(d, i) { return nodes[i].acc; });

      var column = svg.selectAll(".cm_column")
          .data(matrix)
          .enter().append("g")
          .attr("class", "cm_column")
          .attr("transform", function(d, i) { return "translate(" + x(i) + ")rotate(-90)"; });

      column.append("line")
          .attr("x1", -width);

      column.append("text")
          .attr("x", 6)
          .attr("y", x.rangeBand() / 2)
          .attr("dy", ".32em")
          .attr("text-anchor", "start")
          .text(function(d, i) { return nodes[i].name; });

      function row(row) {
        var cell = d3.select(this).selectAll(".cell")
            .data(row.filter(function(d) { return 1; }))
          .enter().append("rect")
            .attr("class", "cell")
            .attr("x", function(d) { return x(d.x); })
            .attr("width", x.rangeBand())
            .attr("height", x.rangeBand())
            .style("fill", function(d) { return color(d.z*100/nodes[d.y].count); })
            .on("mouseover", mouseover)
            .on("mouseout", mouseout);
      }

      function mouseover(p) {
        d3.selectAll(".cm_row_acc text").classed("active", function(d, i) { return i == p.y });
        d3.selectAll(".cm_row text").classed("active", function(d, i) { return i == p.y });
        d3.selectAll(".cm_column text").classed("active", function(d, i) { return i == p.x; });
      }

      function mouseout() {
        d3.selectAll("text").classed("active", false);
      }

      d3.select("#cm_order").on("change", function() {
        order(this.value);
      });

      function order(value) {
        x.domain(orders[value]);

        var t = svg.transition().duration(400);

        t.selectAll(".cm_row")
            .delay(function(d, i) { return x(i) * 4; })
            .attr("transform", function(d, i) { return "translate(0," + x(i) + ")"; }) //Translates the rows
          .selectAll(".cell")
            .delay(function(d) { return x(d.x) * 4; })
            .attr("x", function(d) { return x(d.x); }); //Translates the columns

        t.selectAll(".cm_column") //Horizontal translation for the column labels
            .delay(function(d, i) { return x(i) * 4; })
            .attr("transform", function(d, i) { return "translate(" + x(i) + ")rotate(-90)"; });


        t.selectAll(".cm_row_acc") //Vertical translation for the row-accuracy labels
            .delay(function(d, i) { return x(i) * 4; })
            .attr("transform", function(d, i) { return "translate(0," + x(i) + ")"; });
      }

      //d3.select("#cm_epoch").on("change", function() {
      //  changeEpoch(this.value);
      //});
      //function changeEpoch(value) {
      //  console.log("epoch()");
      //  reloadData(value);
      //  //var t = svg.transition().duration(400);
      //  svg.selectAll(".cm_row_acc text")
      //      .text(function(d, i) { return nodes[i].acc; });
//
      //  //t.selectAll(".cm_row")
      //  //    .text(function(d, i) { console.log(i) });
      //  svg.selectAll(".cm_row")
      //    .data(matrix);
//
      //  svg.selectAll(".cm_column")
      //    .data(matrix);
//
      //  svg.selectAll(".cm_row_acc")
      //    .data(matrix);
//
      //    console.log(matrix);
      //}



  }

  d3.select("#cm_epoch").on("input", function() {
    refreshGraph(cm_data, this.value);
  });

  //Render and set to default epoch
  refreshGraph(cm_data, n_epoch-1);
  d3.select("#cm_epoch")
    .attr("value",n_epoch-1)
    .attr("max",n_epoch-1);

  //var timeout = setTimeout(function() {
  ////  order("group");
  ////  d3.select("#order").property("selectedIndex", 2).node().focus();
  //    //refreshGraph(cm_data,0);
  //    console.log('Ding.');
  //}, 5000);
    

}

