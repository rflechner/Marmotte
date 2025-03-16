namespace Marmotte

open System
open System.Text.Json
open System.Threading
open Microsoft.AspNetCore.Mvc
open Microsoft.OpenApi.Models
open Microsoft.AspNetCore.OpenApi
open System.Reflection

module OpenApiExtensions =

    type SchemaTransformer() =
        
        let removeHttpContextFromSchema (schema:OpenApiSchema) (context:OpenApiSchemaTransformerContext) =
            let typ = context.JsonTypeInfo.Type
            if typ <> null && schema.Properties <> null
            then
                typ.GetProperties(BindingFlags.Public ||| BindingFlags.Instance)
                |> Array.filter (fun p -> p.GetCustomAttribute<FromServicesAttribute>() <> null)
                |> Array.iter (fun p ->
                    p.Name
                    |> JsonNamingPolicy.CamelCase.ConvertName
                    |> schema.Properties.Remove
                    |> ignore
                )
        
        let removeQueryFromSchema (schema:OpenApiSchema) (context:OpenApiSchemaTransformerContext) =
            let typ = context.JsonTypeInfo.Type
            if typ <> null && schema.Properties <> null
            then
                typ.GetProperties(BindingFlags.Public ||| BindingFlags.Instance)
                |> Array.filter (fun p -> p.GetCustomAttribute<FromQueryAttribute>() <> null)
                |> Array.iter (fun p ->
                    p.Name
                    |> JsonNamingPolicy.CamelCase.ConvertName
                    |> schema.Properties.Remove
                    |> ignore
                )

        interface IOpenApiSchemaTransformer with
            member _.TransformAsync(schema:OpenApiSchema, context:OpenApiSchemaTransformerContext, _:CancellationToken) =
                task {
                    removeHttpContextFromSchema schema context
                    removeQueryFromSchema schema context
                    
                    return schema
                }

    type OperationTransformer() =
        
        let mapClrTypeToOpenApiType t =
            if t = typeof<string> then ("string", null)
            elif t = typeof<int> then ("integer", "int32")
            elif t = typeof<int64> then "integer", "int64"
            elif t = typeof<float32> then ("number", "float")
            elif t = typeof<float> then ("number", "double")
            elif t = typeof<bool> then ("boolean", null)
            elif t = typeof<DateTime> then ("string", "date-time")
            elif t.IsEnum then ("string", null)
            else ("object", null)

        let removeQueryFromOperation (operation:OpenApiOperation) (context:OpenApiOperationTransformerContext) =
            for parameter in context.Description.ParameterDescriptions do
                if parameter.Type <> null
                then
                    parameter.Type.GetProperties(BindingFlags.Public ||| BindingFlags.Instance)
                    |> Array.filter (fun p -> p.GetCustomAttribute<FromQueryAttribute>() <> null)
                    |> Array.iter (fun p ->
                        let typeName, format = mapClrTypeToOpenApiType p.PropertyType
                        let param = OpenApiParameter()
                        param.Name <- p.Name
                        param.In <- Nullable ParameterLocation.Query
                        param.Schema <- OpenApiSchema()
                        param.Schema.Type <- typeName
                        param.Schema.Format <- format
                        
                        if operation.Parameters |> isNull
                        then operation.Parameters <- System.Collections.Generic.List<OpenApiParameter>()
                        
                        if operation.Parameters |> Seq.exists (fun i -> i.Name = p.Name) |> not
                        then operation.Parameters.Add param
                    )
            
        interface IOpenApiOperationTransformer with
            member _.TransformAsync(operation:OpenApiOperation, context:OpenApiOperationTransformerContext, _:CancellationToken) =
                task {
                    removeQueryFromOperation operation context
                }
    
    let configureOpenApi (options: OpenApiOptions) =
        options.AddSchemaTransformer<SchemaTransformer>() |> ignore
        options.AddOperationTransformer<OperationTransformer>() |> ignore
    