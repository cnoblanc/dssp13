for folder in $(ls -d */);
do
        echo "Packaging project: "${folder%%/};
        (cd ./$folder; sbt package)
        echo "**********************************"
done

