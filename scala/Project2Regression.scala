import scala.io.Source
import scalation.mathstat.{MatrixD, VectorD}
import scalation.modeling.Regression

object Project2Regression:

  def loadNumericCSV(path: String, targetCol: Int, dropCols: Set[Int] = Set()): (MatrixD, VectorD, Array[String]) =
    val src = Source.fromFile(path)
    val lines = try src.getLines().toArray finally src.close()
    val header = lines.head.split(",").map(_.trim)
    val rows = lines.tail.map(_.split(",").map(_.trim))
    val predictorCols = header.indices.filter(i => i != targetCol && !dropCols.contains(i)).toArray
    val xArray = rows.map { row => Array(1.0) ++ predictorCols.map(i => row(i).toDouble) }
    val yArray = rows.map(row => row(targetCol).toDouble)
    val names = Array("intercept") ++ predictorCols.map(header(_))
    (MatrixD(xArray), VectorD(yArray), names)

  def run(name: String, path: String, targetCol: Int, dropCols: Set[Int] = Set()): Unit =
    val (x, y, names) = loadNumericCSV(path, targetCol, dropCols)
    val model = new Regression(x, y, names)
    model.train(x, y)
    val (_, qof) = model.test(x, y)
    println("=" * 80)
    println(name)
    println(s"features = ${names.mkString(", ")}")
    println(s"parameters = ${model.parameter}")
    println(model.report(qof))
    println(model.summary(x, names, model.parameter.asInstanceOf[VectorD], model.vif()))

  @main def runProject2Regression(): Unit =
    run("AutoMPG", "data/auto_mpg_numeric.csv", 0)
    run("Airfoil", "data/airfoil.csv", 5)
    run("Concrete", "data/concrete.csv", 8)
